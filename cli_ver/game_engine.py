# game_engine.py
import duckdb
import pandas as pd
import numpy as np
import hdbscan
from functools import lru_cache

import database

def get_r_matrix(topic_id: str) -> pd.DataFrame:
    """Generates the user-comment vote matrix (R-matrix) for a specific topic."""
    with duckdb.connect(database=database.DB_PATH, read_only=True) as con:
        query = """
            SELECT v.user_id, v.comment_id, v.vote_type
            FROM votes v
            JOIN comments c ON v.comment_id = c.id
            WHERE c.topic_id = ?
        """
        votes_df = con.execute(query, [topic_id]).fetchdf()

    if votes_df.empty:
        return pd.DataFrame()

    vote_mapping = {'agree': 1, 'neutral': 0, 'disagree': -1}
    votes_df['vote_value'] = votes_df['vote_type'].map(vote_mapping)

    r_matrix = votes_df.pivot_table(
        index='user_id',
        columns='comment_id',
        values='vote_value'
    )
    return r_matrix

def get_clusters(r_matrix: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Performs HDBSCAN clustering on the R-matrix."""
    if r_matrix.empty or len(r_matrix) < 2:
        return np.array([]), {}

    # Fill NaN with a value outside the -1, 0, 1 range to ensure they don't match
    # A common practice is to use a neutral value or a distinct one. Let's use 0 (neutral).
    r_matrix_filled = r_matrix.fillna(0)

    clusterer = hdbscan.HDBSCAN(
        metric='hamming',
        min_cluster_size=max(2, int(len(r_matrix) * 0.1)), # Cluster size is at least 2 or 10% of users
        allow_single_cluster=True,
        min_samples=1
    )
    clusterer.fit(r_matrix_filled.values)
    user_id_to_index = {user_id: i for i, user_id in enumerate(r_matrix.index)}
    return clusterer.labels_, user_id_to_index

@lru_cache(maxsize=128)
def get_user_cluster_info(user_id: str, topic_id: str) -> tuple[int | None, list[str]]:
    """Gets the cluster label and list of users in the same cluster for a specific user."""
    r_matrix = get_r_matrix(topic_id)
    if r_matrix.empty or user_id not in r_matrix.index:
        return None, []

    labels, user_to_idx = get_clusters(r_matrix)
    if not labels.any():
        return 0, list(r_matrix.index) # Everyone is in cluster 0 if clustering fails

    user_index = user_to_idx.get(user_id)
    if user_index is None or user_index >= len(labels):
        return None, []

    target_label = labels[user_index]
    
    # Find all users with the same label
    same_cluster_users = []
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    for i, label in enumerate(labels):
        if label == target_label:
            same_cluster_users.append(idx_to_user[i])

    return int(target_label), same_cluster_users

def get_top_consensus_comments(user_ids: list[str], topic_id: str, k: int = 3) -> list[tuple[str, str]]:
    """Gets top k consensus comments for a list of users."""
    if not user_ids or len(user_ids) < 2:
        return []

    with duckdb.connect(database=database.DB_PATH, read_only=True) as con:
        query = """
            SELECT
                v.comment_id,
                c.content,
                VAR_POP(CASE v.vote_type WHEN 'agree' THEN 1.0 ELSE -1.0 END) as vote_variance
            FROM votes v
            JOIN comments c ON v.comment_id = c.id
            WHERE v.user_id IN (?) AND c.topic_id = ?
            GROUP BY v.comment_id, c.content
            HAVING COUNT(v.user_id) >= 2
            ORDER BY vote_variance ASC
            LIMIT ?
        """
        result = con.execute(query, [user_ids, topic_id, k]).fetchall()
        return [(row[0], row[1]) for row in result]

def name_user_group(user_ids: list[str], topic_id: str) -> tuple[str, str]:
    """Generates a descriptive name for a user group (a 'town')."""
    if not user_ids:
        return "The Void", "An empty space between thoughts."
    
    # This is a simplified version of the naming logic from the Streamlit code
    # A full implementation would require the diversity and significance calculations.
    num_users = len(user_ids)
    if num_users > 10:
        return "Consensus Metropolis", "A large, bustling hub of similar ideas."
    elif num_users > 4:
        return "Agreement Town", "A well-populated settlement where perspectives align."
    elif num_users > 1:
        return "Like-Minded Hamlet", "A small outpost where a few voices echo each other."
    else:
        return "Solitary Post", "A lone perspective in the wilderness."

def get_next_opinion(user_id: str, topic_id: str, current_cluster_users: list[str]) -> tuple[str | None, str | None, bool]:
    """
    The core logic for fetching the next opinion for a user.
    Returns: (comment_id, comment_content, is_repeated_consensus_opinion)
    """
    with duckdb.connect(database=database.DB_PATH, read_only=True) as con:
        # Rule: First, check for top consensus opinions in the player's current town
        # that they have NOT seen yet.
        top_comments = get_top_consensus_comments(current_cluster_users, topic_id, k=5)
        for comment_id, content in top_comments:
            seen_check = con.execute("SELECT 1 FROM seen_comments WHERE user_id = ? AND comment_id = ?", [user_id, comment_id]).fetchone()
            if not seen_check:
                return comment_id, content, False

        # Rule: If all top comments in the new town have been seen, we can show one again.
        # This makes the town feel persistent. We pick one from the top comments.
        if top_comments:
            # Check if this comment was created by the current user
            c_id, c_content = random.choice(top_comments)
            author_check = con.execute("SELECT 1 FROM comments WHERE id = ? AND user_id = ?", [c_id, user_id]).fetchone()
            if not author_check:
                 return c_id, c_content, True

        # Rule: If no consensus or if the only consensus is the user's own comment,
        # get any random comment from the topic that the user has not created and not seen.
        query = """
            SELECT c.id, c.content
            FROM comments c
            WHERE c.topic_id = ?
              AND c.user_id != ?
              AND NOT EXISTS (
                  SELECT 1 FROM seen_comments s
                  WHERE s.comment_id = c.id AND s.user_id = ?
              )
            ORDER BY RANDOM()
            LIMIT 1
        """
        result = con.execute(query, [topic_id, user_id, user_id]).fetchone()
        if result:
            return result[0], result[1], False

    return None, "You've documented every available testimony from this Inquiry. You can still submit your own reports.", False
