import time
import uuid
import random
import urllib.parse # To parse URL parameters
from functools import lru_cache
import streamlit as st
import numpy as np
import pandas as pd
import duckdb
import hdbscan

# Database file path
DB_PATH = '/data/steampolis.duckdb'
DEFAULT_BASE_URL = 'https://huggingface.co/spaces/npc0/SteamPolis/'

# Initialize database tables if they don't exist
def initialize_database():
    try:
        init_con = duckdb.connect(database=DB_PATH, read_only=False)
        init_con.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        init_con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        init_con.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        init_con.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                comment_id TEXT NOT NULL,
                vote_type TEXT NOT NULL CHECK (vote_type IN ('agree', 'disagree', 'neutral')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (comment_id) REFERENCES comments(id),
                UNIQUE (user_id, comment_id)
            )
        """)
        init_con.execute("""
            CREATE TABLE IF NOT EXISTS user_comment_collections (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                comment_id TEXT NOT NULL,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (comment_id) REFERENCES comments(id),
                UNIQUE (user_id, comment_id)
            )
        """)
        init_con.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                last_comment_id_viewed TEXT,
                last_viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                FOREIGN KEY (last_comment_id_viewed) REFERENCES comments(id),
                UNIQUE (user_id, topic_id)
            )
        """)

        # Create system user if it doesn't exist
        try:
            init_con.execute("""
                INSERT INTO users (id, username)
                VALUES ('system', 'System')
                ON CONFLICT (id) DO NOTHING
            """)
        except Exception as e:
            print(f"Warning: Could not create system user: {e}")

    except Exception as e:
        st.error(f"Database initialization failed: {e}")
    finally:
        if 'init_con' in locals() and init_con:
            init_con.close()


def add_example_topic(topic_id, topic_title, topic_description, comments_list):
    """
    Adds an example topic, its comments, and example cluster votes to the database.
    Args:
        topic_id (str): The unique ID for the topic.
        topic_title (str): The title of the topic.
        topic_description (str): The description of the topic.
        comments_list (list): A list of strings, where each string is a comment.
    """
    con = None
    try:
        con = duckdb.connect(database=DB_PATH)

        # Insert the topic - Corrected 'title' to 'name' based on schema
        con.execute("""
            INSERT INTO topics (id, name, description)
            VALUES (?, ?, ?)
            ON CONFLICT (id) DO NOTHING
        """, [topic_id, topic_title, topic_description])

        # --- Add Cluster Users ---
        # Create users who will cast votes, separate from comment authors
        num_users_per_cluster = 5
        cluster1_users = [] # e.g., Pro-Tech supporters
        cluster2_users = [] # e.g., Anti-Tech skeptics
        cluster3_users = [] # e.g., Mixed/Neutral voters

        all_cluster_users = []

        for i in range(num_users_per_cluster):
            user_id = str(uuid.uuid4())
            username = f"cluster1_user_{i+1}_{uuid.uuid4().hex[:4]}@example.com"
            con.execute("INSERT INTO users (id, username) VALUES (?, ?) ON CONFLICT (id) DO NOTHING", [user_id, username])
            cluster1_users.append(user_id)
            all_cluster_users.append(user_id)

            user_id = str(uuid.uuid4())
            username = f"cluster2_user_{i+1}_{uuid.uuid4().hex[:4]}@example.com"
            con.execute("INSERT INTO users (id, username) VALUES (?, ?) ON CONFLICT (id) DO NOTHING", [user_id, username])
            cluster2_users.append(user_id)
            all_cluster_users.append(user_id)

            user_id = str(uuid.uuid4())
            username = f"cluster3_user_{i+1}_{uuid.uuid4().hex[:4]}@example.com"
            con.execute("INSERT INTO users (id, username) VALUES (?, ?) ON CONFLICT (id) DO NOTHING", [user_id, username])
            cluster3_users.append(user_id)
            all_cluster_users.append(user_id)


        # --- Insert comments and associated users ---
        comment_id_map = {} # Map comment text to comment ID
        for comment_text in comments_list:
            comment_id = str(uuid.uuid4())
            # Generate a random user ID and username for the comment author
            author_user_id = str(uuid.uuid4())
            author_username = f"author_{uuid.uuid4().hex[:8]}@example.com"

            # Insert the author user
            con.execute("""
                INSERT INTO users (id, username)
                VALUES (?, ?)
                ON CONFLICT (id) DO NOTHING
            """, [author_user_id, author_username])

            # Insert the comment - Corrected 'text' to 'content' based on schema
            con.execute("""
                INSERT INTO comments (id, topic_id, user_id, content)
                VALUES (?, ?, ?, ?)
            """, [comment_id, topic_id, author_user_id, comment_text])
            comment_id_map[comment_text] = comment_id # Store the mapping

        # --- Add Cluster Votes ---
        # Define comment categories based on the example topic context (Civic Tech Initiative)
        # This is hardcoded based on the context provided in the prompt
        pro_tech_comments = [
            "Finally! A system to track rebel scum more efficiently. This will be a glorious day for the Empire!",
            "Anything that improves the speed of selling junk is good in my book. Maybe I can finally get a decent price for this thermal detonator...",
            "Fascinating! I am programmed to be compliant. I shall analyze this initiative and report my findings to the Emperor.",
            "This is a welcome step towards greater efficiency and transparency... cough... as long as it doesn't affect my personal interests.",
            "As long as it helps me track down my targets, I'm in. The more data, the better.",
            "The Emperor's vision is one of unparalleled order and prosperity! This initiative will usher in a new era of galactic harmony!",
            "I'm interested... Will it help me collect debts more efficiently?",
            "If it improves the entertainment options on Coruscant, I'm all for it.",
            "Another set of orders. Understood, sir!",
            "Excellent... with this, I will have even greater control over the galaxy... cackles maniacally"
        ]
        anti_tech_comments = [
            "This is clearly a data-mining operation. They're going to use it to crush the Rebellion. We need to sabotage it!",
            "The Force guides us to see through their deception. This 'civic tech' will only serve to tighten their grip on the galaxy.",
            "I just want a reliable power converter. Is that too much to ask? This 'civic tech' sounds like more bureaucracy.",
            "I'm already dreading the help desk calls. 'My Death Star won't fire!' 'The Force isn't working!'",
            "This is just a fancy way to track our X-wings. We'll find a way to disable it, just like we did with the Death Star.",
            "Another reason to drown my sorrows in a Jawa Juice. This whole thing stinks of the Empire's incompetence.",
            "This initiative is a waste of resources. We should be focusing on military expansion, not 'civic engagement.'"
        ]
        # Comments not in pro or anti lists are considered neutral/other for this example

        votes_to_insert = []

        # Cluster 1 (Pro-Tech) votes: Agree with pro, Disagree with anti, Mixed/Neutral on others
        for user_id in cluster1_users:
            for comment_text in comments_list:
                comment_id = comment_id_map.get(comment_text)
                if not comment_id: continue

                vote_type = None
                if comment_text in pro_tech_comments:
                    vote_type = 'agree'
                elif comment_text in anti_tech_comments:
                    vote_type = 'disagree'
                else:
                    vote_type = 'neutral'

                if vote_type:
                     # Generate UUID for vote ID and append to list
                     vote_id = str(uuid.uuid4())
                     votes_to_insert.append((vote_id, user_id, comment_id, vote_type))


        # Cluster 2 (Anti-Tech) votes: Disagree with pro, Agree with anti, Mixed/Neutral on others
        for user_id in cluster2_users:
             for comment_text in comments_list:
                comment_id = comment_id_map.get(comment_text)
                if not comment_id: continue

                vote_type = None
                if comment_text in pro_tech_comments:
                    vote_type = 'disagree'
                elif comment_text in anti_tech_comments:
                    vote_type = 'agree'
                else:
                    vote_type = 'neutral'

                if vote_type:
                     # Generate UUID for vote ID and append to list
                     vote_id = str(uuid.uuid4())
                     votes_to_insert.append((vote_id, user_id, comment_id, vote_type))

        # Cluster 3 (Mixed/Neutral) votes: Mostly neutral, some random agree/disagree, many skipped
        for user_id in cluster3_users:
             for comment_text in comments_list:
                comment_id = comment_id_map.get(comment_text)
                if not comment_id: continue

                # Mostly neutral, some random agree/disagree, many skipped
                vote_type = random.choice(['neutral'] * 8 + ['agree', 'disagree'])

                if vote_type:
                     # Generate UUID for vote ID and append to list
                     vote_id = str(uuid.uuid4())
                     votes_to_insert.append((vote_id, user_id, comment_id, vote_type))

        # Insert all collected votes - Added 'id' to the insert statement
        if votes_to_insert:
             con.executemany("""
                 INSERT INTO votes (id, user_id, comment_id, vote_type)
                 VALUES (?, ?, ?, ?)
                 ON CONFLICT (user_id, comment_id) DO NOTHING
             """, votes_to_insert)


        con.commit()
        # print(f"Successfully added topic '{topic_title}', {len(comments_list)} comments, and {len(all_cluster_users)} cluster users with votes.") # Use print for console output
        # st.success(f"Successfully added topic '{topic_title}', {len(comments_list)} comments, and {len(all_cluster_users)} cluster users with votes.") # Use st.success if in Streamlit context

    except Exception as e:
        if con:
            con.rollback()
        print(f"Error adding example topic '{topic_title}' and votes: {e}") # Use print for console output
        # st.error(f"Error adding example topic '{topic_title}' and votes: {e}") # Use st.error if in Streamlit context
    finally:
        if con:
            con.close()


# Example usage (can be called elsewhere, e.g., in an initialization script)
def add_dummy_topic():
    example_topic_id = "15736626"
    example_topic_title = "New Civic Tech Initiative"
    example_topic_description = "Seeker, your mission is to assess the true sentiment regarding the Emperor's new 'Civic Tech' initiative. While officially presented as a means for streamlined governance, enhanced citizen engagement, and a more user-friendly experience, the Emperor requires a candid report on public perception. Gather intelligence on whether citizens view this as a path to order or harbor suspicions of a darker purpose. Report their unfiltered opinions."
    example_comments = [
        "Finally! A system to track rebel scum more efficiently. This will be a glorious day for the Empire!",
        "This is clearly a data-mining operation. They're going to use it to crush the Rebellion. We need to sabotage it!",
        "The Force guides us to see through their deception. This 'civic tech' will only serve to tighten their grip on the galaxy.",
        "As long as it doesn't mess with my profit margins, I'm indifferent. But I suspect it will.",
        "I just want a reliable power converter. Is that too much to ask? This 'civic tech' sounds like more bureaucracy.",
        "Anything that improves the speed of selling junk is good in my book. Maybe I can finally get a decent price for this thermal detonator...",
        "Fascinating! I am programmed to be compliant. I shall analyze this initiative and report my findings to the Emperor.",
        "I'm already dreading the help desk calls. 'My Death Star won't fire!' 'The Force isn't working!'",
        "This is a welcome step towards greater efficiency and transparency... cough... as long as it doesn't affect my personal interests.",
        "This is just a fancy way to track our X-wings. We'll find a way to disable it, just like we did with the Death Star.",
        "As long as it helps me track down my targets, I'm in. The more data, the better.",
        "Another reason to drown my sorrows in a Jawa Juice. This whole thing stinks of the Empire's incompetence.",
        "The Emperor's vision is one of unparalleled order and prosperity! This initiative will usher in a new era of galactic harmony!",
        "Will it have cool spaceships in it? Can I play with it?",
        "Beware the allure of technology. It can be a tool for both good and evil. Trust in the Force, young Padawans.",
        "This initiative is a waste of resources. We should be focusing on military expansion, not 'civic engagement.'",
        "I'm interested... Will it help me collect debts more efficiently?",
        "I'm just trying to survive. This sounds like more trouble than it's worth.",
        "If it improves the entertainment options on Coruscant, I'm all for it.",
        "Another set of orders. Understood, sir!",
        "Excellent... with this, I will have even greater control over the galaxy... cackles maniacally"
    ]
    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=True)
        # Check if the example topic already exists
        result = local_con.execute("SELECT * FROM topics WHERE id = ?", [example_topic_id]).fetchone()
        if result:
            print(f"INFO: Topic '{example_topic_id}' already exists. Skipping dummy data insertion.")
            return # Skip adding dummy data if topic exists
    except Exception as e:
        # Log error but continue, assuming topic doesn't exist if check fails
        print(f"WARNING: Error checking for existing topic '{example_topic_id}': {e}")
        # Don't return, proceed with adding data in case of check error
    finally:
        if local_con:
            local_con.close()
    add_example_topic(example_topic_id, example_topic_title, example_topic_description, example_comments)


def get_ttl_hash(seconds=360):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)

# Helper function to get the R matrix from user voting data
# This matrix represents user-comment interactions (votes)
# Users are rows, comments are columns.
# Values: 1 for 'agree', 0 for 'neutral', -1 for 'disagree', NaN for unvoted.
# Requires pandas and numpy.
def get_r_matrix_from_votes():
    local_con = None
    try:
        # Use read_only=False to maintain consistent configuration across all connections
        local_con = duckdb.connect(database=DB_PATH, read_only=False)

        # Fetch all vote data
        # fetchdf requires pandas
        votes_df = local_con.execute("""
            SELECT user_id, comment_id, vote_type
            FROM votes
        """).fetchdf()

        if votes_df.empty:
            # Return empty matrix and mappings if no votes exist
            # pd.DataFrame requires pandas
            return pd.DataFrame(), {}, {}

        # Map vote types to numerical values
        vote_mapping = {'agree': 1, 'neutral': 0, 'disagree': -1}
        votes_df['vote_value'] = votes_df['vote_type'].map(vote_mapping)

        # Create the R matrix using pivot_table
        # This automatically handles missing user-comment pairs by filling with NaN
        # pivot_table requires pandas
        r_matrix = votes_df.pivot_table(
            index='user_id',
            columns='comment_id',
            values='vote_value'
        )

        # Create mappings from user/comment IDs to matrix indices (optional but useful)
        user_id_to_index = {user_id: i for i, user_id in enumerate(r_matrix.index)}
        comment_id_to_index = {comment_id: i for i, comment_id in enumerate(r_matrix.columns)}

        return r_matrix, user_id_to_index, comment_id_to_index

    except Exception as e:
        # st.error is not available here, just print or log
        print(f"Error generating R matrix: {e}")
        # Return empty results in case of error
        # pd.DataFrame requires pandas
        return pd.DataFrame(), {}, {}
    finally:
        if local_con:
            local_con.close()


# Function to get clusters using HDBSCAN with the custom Hamming distance
# Assumes pandas is imported as pd, numpy as np, and hdbscan is imported
def get_clusters_from_r_matrix(r_matrix):
    """
    Performs HDBSCAN clustering on the R matrix using a custom Hamming-like distance
    that handles NaN values.
    Args:
        r_matrix (pd.DataFrame): The user-comment vote matrix from get_r_matrix_from_votes.
                                 Index should be user_id, columns comment_id.
    Returns:
        np.ndarray: An array of cluster labels for each user in the r_matrix index.
                    -1 indicates noise. Returns empty array if clustering fails or
                    r_matrix is empty.
    """
    # Check if r_matrix is empty
    if r_matrix.empty:
        print("R matrix is empty, cannot perform clustering.")
        return np.array([])

    try:
        # Instantiate HDBSCAN with the custom metric
        # Using default parameters for min_cluster_size and min_samples
        # These might need tuning based on data characteristics and desired cluster granularity
        # allow_single_cluster=True prevents an error if all points form one cluster
        clusterer = hdbscan.HDBSCAN(
            metric='hamming',
            allow_single_cluster=True,
            min_cluster_size=max(int(np.sqrt(len(r_matrix))), 3),
            min_samples=None)

        # Fit the model directly to the DataFrame values
        # HDBSCAN fit expects a numpy array or similar structure
        clusterer.fit(r_matrix.values)

        # Return the cluster labels
        return clusterer.labels_

    except Exception as e:
        # In a Streamlit app context, st.error would be better, but not available here.
        # Print to console/logs.
        print(f"Error during HDBSCAN clustering: {e}")
        return np.array([]) # Return empty array on error


def get_cluster_labels(user_id):
    r_matrix, user_id_to_index, _ = get_r_matrix_from_votes()
    # Check if the user_id exists in the matrix index
    if user_id not in user_id_to_index:
        print(f"Warning: User ID '{user_id}' not found in the R matrix. Cannot perform user-specific filtering for clustering.")
        # Return empty results as filtering based on this user is not possible.
        # The downstream function get_user_cluster_label handles the user not being in the index.
        # Returning empty arrays/dict matches the structure of the expected return value.
        return np.array([]), {} # Return empty labels and empty index map

    # Get the row for the specific user
    user_row = r_matrix.loc[user_id]

    # Find columns where the user has voted (values are not NaN)
    voted_comment_ids = user_row.dropna().index

    # Ensure we handle the case where the user hasn't voted on anything
    if voted_comment_ids.empty:
        print(f"Warning: User ID '{user_id}' has not voted on any comments. Cannot perform clustering based on votes.")
        # If no votes, no columns to cluster on. Return empty results.
        return np.array([]), {}

    # Filter the r_matrix to include only these columns
    # This is the matrix that will be used for clustering in the next step.
    # The subsequent line calling get_clusters_from_r_matrix should use this variable.
    r_matrix = r_matrix[voted_comment_ids]
    cluster_labels = get_clusters_from_r_matrix(r_matrix)
    if len(cluster_labels) == 0:
        cluster_labels = [0] * len(user_id_to_index)
    return cluster_labels, user_id_to_index


# Function to get the cluster label for a specific user
@lru_cache()
def get_user_cluster_label(user_id, ttl_hash=None):
    """
    Gets the HDBSCAN cluster label for a specific user and a list of users
    sharing the same cluster.
    Args:
        user_id (str): The ID of the user.
    Returns:
        tuple: A tuple containing:
            - int or None: The cluster label (an integer, -1 for noise) if the user
                           is found in the clustering result, otherwise None.
            - list[str]: A list of user IDs (including the input user_id if found)
                         that belong to the same cluster. Returns an empty list
                         if the user is not found or has no cluster label.
    """
    # get_cluster_labels is already cached, so calling it repeatedly is fine
    cluster_labels, user_id_to_index = get_cluster_labels(user_id)

    # Create a reverse mapping from index to user_id for easier lookup
    index_to_user_id = {index: uid for uid, index in user_id_to_index.items()}

    target_cluster_label = None
    same_cluster_users = []

    # Check if the user_id exists in the mapping
    if user_id in user_id_to_index:
        user_index = user_id_to_index[user_id]
        # Ensure the index is within the bounds of the cluster_labels array
        if 0 <= user_index < len(cluster_labels):
            target_cluster_label = int(cluster_labels[user_index]) # Get the target label

            # Find all users with the same cluster label
            for index, current_user_id in index_to_user_id.items():
                # Ensure the index is valid for cluster_labels
                if 0 <= index < len(cluster_labels):
                    current_user_label = int(cluster_labels[index])
                    if current_user_label == target_cluster_label:
                        same_cluster_users.append(current_user_id)
                else:
                     # This case should ideally not happen if index_to_user_id is consistent
                     print(f"Warning: Index {index} from index_to_user_id out of bounds for cluster labels array length {len(cluster_labels)}")


        else:
            # This case should ideally not happen if user_id_to_index is consistent
            print(f"Warning: User index {user_index} out of bounds for cluster labels array length {len(cluster_labels)}")
            # Return None and empty list as user couldn't be processed
            return None, []
    else:
        # User not found in the R matrix used for clustering (e.g., new user with no votes)
        # print(f"User ID {user_id} not found in clustering data.") # Optional: for debugging
        # Return None and empty list as user is not part of the current clustering result
        return None, []

    # Return the target user's label and the list of users in that cluster
    return target_cluster_label, same_cluster_users


# Helper function to get top k most polarized comments for a list of users
def get_top_k_consensus_comments_for_users(user_ids, topic_id, k=5):
    """
    Retrieves the top k comments with the highest voting consensus (lowest variance)
    among a given list of users *for a specific topic*.
    Consensus is measured by the population variance (VAR_POP) of numerical
    vote scores (-1 for 'disagree', 0 for 'neutral', 1 for 'agree').
    Lower variance indicates higher consensus.
    Args:
        user_ids (list[str]): A list of user IDs.
        topic_id (str): The ID of the topic to filter comments by.
        k (int): The number of top comments to retrieve.
    Returns:
        list[tuple]: A list of tuples, where each tuple contains
                     (comment_id, comment_content, vote_variance),
                     ordered by vote_variance ascending (lowest variance first).
                     Returns an empty list if no votes are found for these users
                     on this topic, or on error, or if the group has fewer than 2 users.
    """
    if not user_ids or len(user_ids) < 2:
        # Need at least 2 users from the group to calculate meaningful variance
        # print("Warning: get_top_k_consensus_comments_for_users called with fewer than 2 user_ids.") # Optional debug
        return [] # Cannot query without user IDs or with only one user

    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=True)

        # Use parameterized query for the list of user IDs and topic ID
        # DuckDB's Python API handles lists for IN clauses
        query = """
            SELECT
                v.comment_id,
                c.content,
                VAR_POP(CASE
                    WHEN v.vote_type = 'agree' THEN 1.0
                    WHEN v.vote_type = 'neutral' THEN 0.0
                    WHEN v.vote_type = 'disagree' THEN -1.0
                    ELSE NULL -- Should not happen with current data
                END) as vote_variance,
                COUNT(v.user_id) as num_votes_in_group -- Include count for potential tie-breaking
            FROM votes v
            JOIN comments c ON v.comment_id = c.id
            WHERE v.user_id IN (?) AND c.topic_id = ? -- Filter by user IDs and topic ID
            GROUP BY v.comment_id, c.content
            HAVING COUNT(v.user_id) >= 2 -- Ensure at least 2 users from the list voted on this comment
            ORDER BY vote_variance ASC, num_votes_in_group DESC -- Order by lowest variance, then by number of votes (more votes = stronger consensus)
            LIMIT ?
        """
        # Pass the list of user_ids, topic_id, and k as parameters
        # DuckDB requires list parameters to be wrapped in a list/tuple for the execute method
        result = local_con.execute(query, [user_ids, topic_id, k]).fetchall()

        # The result includes comment_id, content, variance, and count.
        # We only need comment_id, content, and variance for the return value as per docstring.
        # The count was used for ordering.
        formatted_result = [(row[0], row[1], row[2]) for row in result]

        return formatted_result

    except Exception as e:
        # st.error is not available here, just print or log
        print(f"Error getting top k consensus comments for users {user_ids} in topic {topic_id}: {e}")
        return [] # Return empty list on error
    finally:
        if local_con:
            local_con.close()


def estimate_group_voting_diversity(user_ids, topic_id):
    """
    Estimates the diversity of voting within a group of users for a specific topic.
    Diversity is measured by the average variance of numerical vote scores (-1, 0, 1)
    across comments that at least two users in the group have voted on.
    Args:
        user_ids (list[str]): A list of user IDs belonging to the group.
        topic_id (str): The ID of the topic.
    Returns:
        float: A diversity score between 0.0 and 1.0. 0.0 indicates no diversity
               (all users voted the same way on all shared comments), 1.0 indicates
               maximum possible diversity (e.g., half agree, half disagree on shared comments).
               Returns 0.0 if the group has less than 2 users or if no comments
               were voted on by at least two users in the group.
    """
    # Convert list to tuple for caching purposes (tuples are hashable)
    user_ids_tuple = tuple(user_ids)

    if not user_ids_tuple or len(user_ids_tuple) < 2:
        return 0.0

    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=True)

        # Get all votes for the given topic by the specified users
        # Join with comments to filter by topic_id
        # Construct the IN clause dynamically to avoid the conversion error
        placeholders = ', '.join(['?'] * len(user_ids_tuple))
        query = f"""
            SELECT
                v.comment_id,
                v.user_id,
                v.vote_type
            FROM votes v
            JOIN comments c ON v.comment_id = c.id
            WHERE c.topic_id = ? AND v.user_id IN ({placeholders})
        """
        # Pass topic_id and then all user_ids as separate parameters
        params = [topic_id] + list(user_ids_tuple) # Combine topic_id and user_ids
        results = local_con.execute(query, params).fetchall()

        if not results:
            return 0.0 # No votes found for this group on this topic

        # Map vote types to numerical scores
        vote_map = {'agree': 1.0, 'neutral': 0.0, 'disagree': -1.0}

        # Group votes by comment ID
        votes_by_comment = {}
        for comment_id, user_id, vote_type in results:
            if comment_id not in votes_by_comment:
                votes_by_comment[comment_id] = []
            # Append the numerical vote score
            votes_by_comment[comment_id].append(vote_map.get(vote_type, 0.0)) # Default to 0.0 for unknown types

        # Calculate variance for comments voted on by at least two users in the group
        variances = []
        for comment_id, comment_votes in votes_by_comment.items():
            # Ensure the comment was voted on by at least two users from the input list
            if len(comment_votes) >= 2:
                # Use numpy to calculate variance
                variances.append(np.var(comment_votes))

        if not variances:
            return 0.0 # No comments voted on by at least two users in the group

        # The maximum possible variance for values in [-1, 0, 1] is 1.0
        # (e.g., half votes are 1, half are -1).
        # The average variance is already in the range [0, 1].
        average_variance = np.mean(variances)

        return average_variance

    except Exception as e:
        # st.error is not available here, just print or log
        print(f"Error estimating group voting diversity for topic {topic_id} and users {user_ids_tuple}: {e}")
        return 0.0 # Return 0.0 on error
    finally:
        if local_con:
            local_con.close()

# Helper function to name a group of users based on their participation and voting diversity
def name_user_group(user_ids, topic_id):
    """
    Generates a descriptive name and description for a group of users within a
    specific topic based on their participation level and voting diversity,
    themed around a seeker's journey through different settlements.
    Args:
        user_ids (list[str]): A list of user IDs belonging to the group.
        topic_id (str): The ID of the topic.
    Returns:
        tuple[str, str]: A tuple containing the name and description for the group.
                         Returns ("The Silent Threshold", "You sense a presence, but no clear voices emerge from this place.")
                         or ("The Silent Valley", "A valley where the inhabitants reside, but their voices are silent on this matter.")
                         or ("The Untrodden Path", "This path has not yet been explored by any travelers.")
                         or ("The Shrouded Keep", "A place hidden by mystery and uncertainty.")
                         in edge cases or on error.
    """
    # Handle empty user list - implies no specific group is being considered, but not necessarily an empty world
    if not user_ids:
        return "The Silent Threshold", "You sense a presence, but no clear voices emerge from this place."

    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=True)

        # 1. Get total unique users who voted in the topic
        # Specify v.user_id to avoid ambiguity
        total_voters_result = local_con.execute("""
            SELECT COUNT(DISTINCT v.user_id)
            FROM votes v
            JOIN comments c ON v.comment_id = c.id
            WHERE c.topic_id = ?
        """, [topic_id]).fetchone()
        total_voters_in_topic = total_voters_result[0] if total_voters_result else 0

        # 2. Get unique users from the input list who voted in the topic
        # Filter user_ids to only those present in the votes table for this topic
        # Construct the IN clause dynamically to avoid casting issues
        # This part correctly uses the list by expanding placeholders
        placeholders = ', '.join(['?'] * len(user_ids))
        group_voters_query = f"""
            SELECT COUNT(DISTINCT v.user_id)
            FROM votes v
            JOIN comments c ON v.comment_id = c.id
            WHERE c.topic_id = ? AND v.user_id IN ({placeholders})
        """
        # Pass topic_id and then all user_ids as separate parameters
        group_voters_result = local_con.execute(group_voters_query, [topic_id] + user_ids).fetchone()
        group_voters_count = group_voters_result[0] if group_voters_result else 0

        # Handle case where no one in the group has voted on this topic
        # The group members exist, but are silent on this specific matter
        if group_voters_count == 0:
             return "The Silent Valley", "A valley where the inhabitants reside, but their voices are silent on this matter."

        # Handle case where topic has no voters at all (the path hasn't been explored)
        if total_voters_in_topic == 0:
             # This case is unlikely if group_voters_count > 0, but for safety
             return "The Untrodden Path", "This path has not yet been explored by any travelers."


        # 3. Calculate significance (proportion of group voters among all topic voters)
        significance_proportion = group_voters_count / total_voters_in_topic

        # 4. Get diversity score for the group
        diversity_score = estimate_group_voting_diversity(user_ids, topic_id)

        # 5. Determine name and description based on significance and diversity
        # Define thresholds (can be tuned)
        SIG_LOW_THRESHOLD = 0.1
        SIG_MED_THRESHOLD = 0.5 # High if > MED, Med if > LOW and <= MED, Low if <= LOW
        DIV_LOW_THRESHOLD = 0.2
        DIV_MED_THRESHOLD = 0.5 # High if > MED, Med if > LOW and <= MED, Low if <= LOW

        significance_level = "low"
        if significance_proportion > SIG_MED_THRESHOLD:
            significance_level = "high"
        elif significance_proportion > SIG_LOW_THRESHOLD:
            significance_level = "medium"

        diversity_level = "low"
        if diversity_score > DIV_MED_THRESHOLD:
            diversity_level = "high"
        elif diversity_score > DIV_LOW_THRESHOLD:
            diversity_level = "medium"

        # Assign names and descriptions based on levels (Themed for seeker quest)
        if significance_level == "high":
            if diversity_level == "low":
                return "The Village of Unity", "A large settlement where the inhabitants share a common, strong belief."
            elif diversity_level == "medium":
                return "The Town of Common Ground", "A bustling town where most agree, though minor differences are acknowledged."
            else: # high diversity
                return "The City of Many Voices", "A major city where travelers from all paths meet and share a wide array of perspectives."
        elif significance_level == "medium":
            if diversity_level == "low":
                return "The Hamlet of Quiet Accord", "A peaceful hamlet of moderate size where opinions align with little dissent."
            elif diversity_level == "medium":
                return "The Crossroads Inn", "An inn of moderate size where various travelers pause, sharing somewhat varied views."
            else: # high diversity
                return "The Debating Circle", "A gathering place of moderate size known for its spirited and diverse discussions."
        else: # low significance
            if diversity_level == "low":
                return "The Like-Minded Few", "A small, isolated group whose thoughts resonate closely."
            elif diversity_level == "medium":
                return "The Scattered Camps", "A few scattered camps where different, quiet thoughts reside."
            else: # high diversity
                return "The Whispering Caves", "A small, hidden network of caves where many different ideas are shared in hushed tones."

    except Exception as e:
        st.error(f"Error naming user group for topic {topic_id} and users {user_ids}: {e}")
        return "The Shrouded Keep", "A place hidden by mystery and uncertainty." # Default name and description on error
    finally:
        if local_con:
            local_con.close()


# Helper function to get a random unvoted comment
def get_random_unvoted_comment(user_id, topic_id):
    new_area_comments = st.session_state.get("_new_area_comments", [])
    if len(new_area_comments) != 0:
        value = new_area_comments.pop()
        st.session_state._new_area_comments = new_area_comments
        return value[0], value[1]
    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=False)

        # First, check if there are any comments at all in the topic
        comment_count = local_con.execute("""
            SELECT COUNT(*) FROM comments WHERE topic_id = ?
        """, [topic_id]).fetchone()[0]

        if comment_count == 0:
            return None, "Share your insight!"

        # Attempt to get a random comment that the user has NOT voted on
        result = local_con.execute("""
            SELECT c.id, c.content
            FROM comments c
            WHERE c.topic_id = ?
            AND NOT EXISTS (
                SELECT * FROM votes v
                WHERE v.comment_id = c.id AND v.user_id = ?
            )
            ORDER BY RANDOM()
            LIMIT 1
        """, [topic_id, user_id]).fetchone()
        if result:
            # Check for cluster change and set message flag
            current_label, current_users = get_user_cluster_label(user_id, get_ttl_hash(10))
            current_users_set = set(current_users)

            previous_label = st.session_state.get('_previous_cluster_label')
            previous_users_set = st.session_state.get('_previous_cluster_users_set', set())

            # Check if cluster label has changed AND the set of users in the new cluster is different
            # This indicates the user has moved to a different group of commenters
            if current_label is not None and previous_label is not None and current_label != previous_label:
                # Calculate overlap (Jaccard Index)
                intersection_size = len(current_users_set.intersection(previous_users_set))
                union_size = len(current_users_set.union(previous_users_set))

                # Check if overlap (Jaccard) is over 70%
                # Handle case where union_size is 0 (both sets empty)
                if union_size > 0 and (intersection_size / union_size) > 0.7:
                    # Set a flag in session state to display the message later in the main rendering logic
                    st.session_state._show_new_area_message = True
                    # Fetch comments from the NEW area (current_users_set)
                    # Note: get_top_k_consensus_comments_for_users expects a list, not a set
                    new_area_comments = get_top_k_consensus_comments_for_users(list(current_users_set), topic_id, k=5)
                    st.session_state._new_area_comments = new_area_comments
                    # print(f"DEBUG: Cluster changed for user {user_id} in topic {topic_id}: {previous_label} -> {current_label}")
                    # print(f"DEBUG: Previous users count: {len(previous_users_set)}, Current users count: {len(current_users_set)}")
            st.session_state._previous_cluster_label = current_label
            st.session_state._previous_cluster_users_set = current_users_set

            # Found an unvoted comment
            return result[0], result[1]
        else:
            # No unvoted comments found for this user in this topic
            return None, "No new thoughts for now..."

    except Exception as e:
        st.error(f"Error getting random unvoted comment: {e}")
        return None, f"Error loading comments: {str(e)}"
    finally:
        if local_con:
            local_con.close()

# Helper function to find or create a user
def find_or_create_user(username):
    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=False)
        user_result = local_con.execute("SELECT id FROM users WHERE username = ?", [username]).fetchone()
        if user_result:
            return user_result[0]
        else:
            user_id = str(uuid.uuid4())
            local_con.execute("INSERT INTO users (id, username) VALUES (?, ?)", [user_id, username])
            return user_id
    except Exception as e:
        st.error(f"Error finding or creating user: {e}")
        return None
    finally:
        if local_con:
            local_con.close()

# Helper function to update user progress
def update_user_progress(user_id, topic_id, comment_id):
    local_con = None
    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=False)
        progress_id = str(uuid.uuid4())
        local_con.execute("""
            INSERT INTO user_progress (id, user_id, topic_id, last_comment_id_viewed) VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id, topic_id) DO UPDATE SET
                last_comment_id_viewed = EXCLUDED.last_comment_id_viewed
        """, [progress_id, user_id, topic_id, comment_id])
    except Exception as e:
        st.error(f"Error updating user progress: {e}")
    finally:
        if local_con:
            local_con.close()

# --- Page Functions ---

def home_page():
    st.title("Welcome to SteamPolis")
    st.markdown("Choose an option:")

    if st.button("Create New Topic (Quest)"):
        st.session_state.page = 'create_topic'
        st.rerun()

    st.markdown("---")
    st.markdown("Or join an existing topic (quest):")
    topic_input = st.text_input("Enter Topic ID or URL")

    if st.button("Join Topic"):
        topic_id = topic_input.strip()
        if topic_id.startswith('http'): # Handle full URL
             parsed_url = urllib.parse.urlparse(topic_id)
             query_params = urllib.parse.parse_qs(parsed_url.query)
             topic_id = query_params.get('topic', [None])[0]

        if topic_id:
            st.session_state.page = 'view_topic'
            st.session_state.current_topic_id = topic_id
            # Attempt to load email from session state (mimics browser state)
            # If email exists, handle email submission logic immediately on view page load
            st.rerun()
        else:
            st.warning("Please enter a valid Topic ID or URL.")


def create_topic_page():
    st.title("Create a New Topic")

    new_topic_name = st.text_input("Topic Name (Imagine you are the king, how would you share your concern)")
    new_topic_description = st.text_area('Description (Begin with "I want to figure out...", imagine you are the king, what would you want to know)', height=150)
    new_topic_seed_comments = st.text_area("Initial Comments (separate by new line, imagine there are civilians what will they answer)", height=200)
    creator_email = st.text_input("Enter your Email (required for creation)")

    if st.button("Create Topic"):
        if not creator_email:
            st.error("Email is required to create a topic.")
            return

        topic_id = str(uuid.uuid4())[:8]
        user_id = find_or_create_user(creator_email)

        if user_id:
            local_con = None
            try:
                local_con = duckdb.connect(database=DB_PATH, read_only=False)
                local_con.execute("INSERT INTO topics (id, name, description) VALUES (?, ?, ?)", [topic_id, new_topic_name, new_topic_description])

                seed_comments = [c.strip() for c in new_topic_seed_comments.split('\n') if c.strip()]
                for comment in seed_comments:
                    comment_id = str(uuid.uuid4())
                    local_con.execute("INSERT INTO comments (id, topic_id, user_id, content) VALUES (?, ?, ?, ?)",
                                      [comment_id, topic_id, 'system', comment])

                # Get the first comment to display after creation
                comment_to_display_id, comment_to_display_content = get_random_unvoted_comment(user_id, topic_id)

                # Set initial progress for creator
                update_user_progress(user_id, topic_id, comment_to_display_id)

                st.session_state.page = 'view_topic'
                st.session_state.current_topic_id = topic_id
                st.session_state.user_email = creator_email # Store email in session state
                st.session_state.current_comment_id = comment_to_display_id
                st.session_state.current_comment_content = comment_to_display_content
                st.session_state.comment_history = ""

                st.success(f"Topic '{new_topic_name}' created!")
                st.rerun()

            except Exception as e:
                st.error(f"Error creating topic: {e}")
            finally:
                if local_con:
                    local_con.close()
        else:
            st.error("Could not find or create user.")


    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.rerun()


def view_topic_page():
    topic_id = st.session_state.get('current_topic_id')
    user_email = st.session_state.get('user_email', '')
    current_comment_id = st.session_state.get('current_comment_id')
    current_comment_content = st.session_state.get('current_comment_content', "Loading comments...")
    comment_history = st.session_state.get('comment_history', "")

    if not topic_id:
        st.warning("No topic selected. Returning to home.")
        st.session_state.page = 'home'
        st.rerun()
        return

    local_con = None
    topic_name = "Loading..."
    topic_description = "Loading..."

    try:
        local_con = duckdb.connect(database=DB_PATH, read_only=True)
        topic_result = local_con.execute("SELECT name, description FROM topics WHERE id = ?", [topic_id]).fetchone()
        if topic_result:
            topic_name, topic_description = topic_result
        else:
            st.error(f"Topic ID '{topic_id}' not found.")
            st.session_state.page = 'home'
            st.rerun()
            return
    except Exception as e:
        st.error(f"Error loading topic details: {e}")
        if local_con:
             local_con.close()
        st.session_state.page = 'home'
        st.rerun()
        return
    finally:
        if local_con:
            local_con.close()


    # Include functional information
    st.markdown(f"**Shareable Quest Scroll ID:** `{topic_id}`")
    # Construct shareable link using current app URL
    app_url = st.query_params.get('base', [DEFAULT_BASE_URL])[0] # Get base URL if available
    shareable_link = f"{app_url}?topic={topic_id}" if app_url else f"?topic={topic_id}"
    st.markdown(f"**Shareable Scroll Link:** `{shareable_link}`")

    st.title("Seeker Quest")

    # Check if user email is available in session state.
    # user_email is already retrieved from st.session_state at the start of view_topic_page.
    if user_email:
        # Get the user ID. find_or_create_user handles the DB connection internally.
        user_id = find_or_create_user(user_email)
        if user_id:
            # Check if user has any progress recorded for this specific topic.
            # This indicates they have viewed comments or interacted before.
            local_con = None
            progress_exists = False
            try:
                local_con = duckdb.connect(database=DB_PATH, read_only=True)

                # Check if the user has voted on any comment in this topic
                # This requires joining votes with comments to filter by topic_id
                voted_result = local_con.execute("""
                    SELECT 1
                    FROM votes v
                    JOIN comments c ON v.comment_id = c.id
                    WHERE v.user_id = ? AND c.topic_id = ?
                    LIMIT 1
                """, [user_id, topic_id]).fetchone()

                # Check if the user has submitted any comment in this topic
                commented_result = local_con.execute("""
                    SELECT 1
                    FROM comments
                    WHERE user_id = ? AND topic_id = ?
                    LIMIT 1
                """, [user_id, topic_id]).fetchone()

                # Progress exists if the user has either voted or commented in this topic
                progress_exists = voted_result is not None or commented_result is not None
            except Exception as e:
                # Log error but don't stop the app. Assume no progress on error.
                st.error(f"Error checking user progress for greeting: {e}")
                # progress_exists remains False
            finally:
                if local_con:
                    local_con.close()

            # Display the appropriate greeting based on progress
            if progress_exists:
                # Acknowledge return and remind of quest
                st.markdown("Welcome back, Seeker. Your journey through the whispers of Aethelgard continues.")
                st.markdown(f"You pause to recall the heart of the Emperor's concern regarding **{topic_name}**: `{topic_description}`.")

                # Introduce the next comment
                st.markdown("As you press onward, you encounter another soul willing to share their thoughts on this vital matter.")
            else:
                # Introduce the setting and the Emperor's concern
                st.markdown("Welcome, Seeker, to the ancient Kingdom of Aethelgard, a realm of digital whispers and forgotten wisdom.")
                st.markdown("For centuries, Aethelgard has stood, preserving the echoes of an age long past. But now, a matter of great weight troubles the Emperor's thoughts.")
                st.markdown(f"The Emperor seeks clarity on a crucial topic: **`{topic_name}`**.")

                # Explain the quest and the user's role
                st.markdown("You, among a select few, have been summoned for a vital quest: to traverse the kingdom, gather insights, and illuminate this matter for the throne.")
                st.markdown(f"At a recent royal gathering, the Emperor revealed the heart of their concern, the very essence of your mission: `{topic_description}`")

                # Transition to the task
                st.markdown("Your journey begins now. The path leads to the first village, where the voices of the realm await your ear.")


    # --- Email Prompt ---
    if not user_email:
        st.subheader("Enter your Email to view comments and progress")
        view_user_email_input = st.text_input("Your Email", key="view_email_input")
        if st.button("Submit Email", key="submit_view_email"):
            if view_user_email_input:
                st.session_state.user_email = view_user_email_input
                user_id = find_or_create_user(view_user_email_input)
                if user_id:
                    comment_to_display_id, comment_to_display_content = get_random_unvoted_comment(user_id, topic_id)
                    st.session_state.current_comment_id = comment_to_display_id
                    st.session_state.current_comment_content = comment_to_display_content
                    update_user_progress(user_id, topic_id, comment_to_display_id)
                    st.session_state.comment_history = "" # Reset history on new email submission
                    st.rerun()
                else:
                    st.error("Could not find or create user with that email.")
            else:
                st.warning("Please enter your email.")
        return # Stop rendering the rest until email is submitted

    # --- Comment Display and Voting ---
    # Define introductory phrases for encountering a new perspective
    intro_phrases = [
        "A new whisper reaches your ear",
        "You ponder a fresh perspective",
        "Another voice shares their view",
        "A thought emerges from the crowd",
        "The wind carries a new idea",
        "Someone offers an insight",
        "You overhear a comment",
        "A different angle appears",
        "The village elder shares",
        "A traveler murmurs",
    ]
    # Randomly select a phrase
    random_phrase = random.choice(intro_phrases)
    st.markdown(comment_history)

    if current_comment_id: # Only show voting if there's a comment to vote on
        # Display comment history and the current comment with the random intro
        if st.session_state.get('_show_new_area_message', True):
            st.session_state._show_new_area_message = False
            _, user_ids = get_user_cluster_label(user_id, get_ttl_hash(10))
            new_area_name, desc = name_user_group(user_ids, topic_id)
            for statm in [
                f"📚 You've collected **{len(comment_history.splitlines())}** insights this time.",
                f"🗺️ And yet, your journey leads you to a new place: **{new_area_name}**! {desc}"]:
                st.markdown(statm)
                st.session_state.comment_history += f"\n\n{statm}"
        st.markdown(f"[Collected new insight, {random_phrase}]:\n* {current_comment_content}")

        # Handle vote logic
        def handle_vote(vote_type, comment_id, topic_id, user_id):
            # Add JavaScript to scroll to the bottom anchor after the page reloads
            # This script will be included in the next render cycle triggered by st.rerun()
            # Ensure an element with id="bottom" exists in the rendered page,
            # typically placed after the content you want to scroll to (e.g., comment history).
            local_con = None
            try:
                local_con = duckdb.connect(database=DB_PATH, read_only=False)
                # Use INSERT OR REPLACE INTO or ON CONFLICT DO UPDATE to handle repeat votes
                # The UNIQUE constraint on (user_id, comment_id) in the votes table
                # allows us to update the existing vote if one already exists for this user/comment pair.
                # We generate a new UUID for the 'id' column, but it will only be used
                # if this is a new insert. If it's an update, the existing 'id' is kept.
                vote_id = str(uuid.uuid4()) # Generate a new UUID for the potential insert
                local_con.execute("""
                    INSERT INTO votes (id, user_id, comment_id, vote_type)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (user_id, comment_id)
                    DO UPDATE SET
                        vote_type = excluded.vote_type, -- Update vote_type with the new value
                        created_at = current_localtimestamp(); -- Update timestamp to reflect the latest vote
                """, [vote_id, user_id, comment_id, vote_type])

                # Append voted comment to history
                # Note: This appends the comment regardless of whether it was a new vote or an update.
                # The history is a simple log, not a reflection of vote changes.
                vote_text = "👍" if vote_type == "agree" else "👎" if vote_type == "disagree" else "😐"
                comment_history = st.session_state.comment_history.split("\n\n")
                if len(comment_history) > 10:
                    comment_history = ["..."] + comment_history[-10:]
                st.session_state.comment_history = "\n\n".join(comment_history)
                st.session_state.comment_history += f"\n\n{vote_text} {current_comment_content}"

                # Check vote count and trigger special event
                # Initialize vote_count if it doesn't exist
                if 'vote_count' not in st.session_state:
                    st.session_state.vote_count = 0

                # Increment vote count only if it was a new vote or a change?
                # The current logic increments on every button click. Let's keep that for now
                # as it drives the special event trigger based on interaction frequency.
                st.session_state.vote_count += 1

                # Check if it's time for a potential special event (every 5 votes/interactions)
                if st.session_state.vote_count % 5 == 0:
                    st.session_state.vote_count = 0 # Reset count after triggering
                    # 30% chance to trigger the special sharing event
                    if random.random() < 0.3:
                        prompts = [
                            "An elder approaches you, seeking your perspective on the Emperor's concern. What wisdom do you share?",
                            "A letter arrives from the Emperor's office, requesting your personal insight on the matter. What counsel do you offer?",
                            "As you walk through the streets, people gather, eager to hear your thoughts on the Emperor's dilemma. What advice do you give?"
                        ]
                        # Pass the current topic_id to share_wisdom if needed, though it's not currently used there.
                        st.markdown(random.choice(prompts))
                        new_comment_text = st.text_area("Your Insight that different from others above (Empty to skip)", key="tmp_new_comment_input")
                        st.session_state.handling_vote = True # lock
                        if st.button("Share Wisdom"):
                            if new_comment_text and len(new_comment_text.strip()):
                                user_email = st.session_state.get('user_email', '')
                                user_id = find_or_create_user(user_email) # Ensure user exists
                                if user_id:
                                    local_con = None
                                    try:
                                        local_con = duckdb.connect(database=DB_PATH, read_only=False)
                                        comment_id = str(uuid.uuid4())
                                        local_con.execute("""
                                            INSERT INTO comments (id, topic_id, user_id, content)
                                            VALUES (?, ?, ?, ?)
                                        """, [comment_id, topic_id, user_id, new_comment_text])

                                        # Append new comment to history
                                        st.session_state.comment_history += f"\n\n💬 {new_comment_text}"

                                        st.session_state.tmp_new_comment_input = "" # Clear input box
                                    except Exception as e:
                                        st.error(f"Error sharing information: {e}")
                                    finally:
                                        if local_con:
                                            local_con.close()
                                else:
                                    st.error("Could not find or create user.")
                            st.session_state.handling_vote = False # lock

                # Get next comment
                # This should always get the next unvoted comment for the user in this topic.
                next_comment_id, next_comment_content = get_random_unvoted_comment(user_id, topic_id)
                st.session_state.current_comment_id = next_comment_id
                st.session_state.current_comment_content = next_comment_content

                # Update progress
                # Update the user's progress to the next comment they should see.
                update_user_progress(user_id, topic_id, next_comment_id)

                st.session_state._voting_in_progress = False
                if st.session_state.get("handling_vote", False) is False:
                    st.rerun() # Rerun to update UI

            except Exception as e:
                st.error(f"Error processing vote: {e}")
            finally:
                if local_con:
                    local_con.close()

        st.session_state._voting_in_progress = st.session_state.get("_voting_in_progress", False)
        col1, col2, col3, col4 = st.columns(4)
        user_id = find_or_create_user(user_email) # Ensure user exists

        col1.markdown("*Personally I...*")
        if col2.button("Agree", disabled=st.session_state.get('_voting_in_progress', False)):
            # Set a flag immediately to disable buttons until next render
            if st.session_state._voting_in_progress == False:
                st.session_state._voting_in_progress = True
                handle_vote("agree", current_comment_id, topic_id, user_id)
        if col3.button("Neutral", disabled=st.session_state.get('_voting_in_progress', False)):
            # Set a flag immediately to disable buttons until next render
            if st.session_state._voting_in_progress == False:
                st.session_state._voting_in_progress = True
                handle_vote("neutral", current_comment_id, topic_id, user_id)
        if col4.button("Disagree", disabled=st.session_state.get('_voting_in_progress', False)):
            # Set a flag immediately to disable buttons until next render
            if st.session_state._voting_in_progress == False:
                st.session_state._voting_in_progress = True
                handle_vote("disagree", current_comment_id, topic_id, user_id)

    else:
        st.info("No more comments to vote on in this topic." if "No more comments" in current_comment_content else current_comment_content)


    st.markdown("")

    # --- Comment Submission ---
    with st.expander("Offer Your Counsel to the Emperor", expanded=False):
        st.markdown("Having heard the thoughts of others, what wisdom do you wish to share regarding the Emperor's concern?")
        new_comment_text = st.text_area(f"Your Insight", key="new_comment_input")
        if st.button("Share Your Wisdom"):
            if new_comment_text:
                user_email = st.session_state.get('user_email', '')
                user_id = find_or_create_user(user_email) # Ensure user exists
                if user_id:
                    local_con = None
                    try:
                        local_con = duckdb.connect(database=DB_PATH, read_only=False)
                        comment_id = str(uuid.uuid4())
                        local_con.execute("""
                            INSERT INTO comments (id, topic_id, user_id, content)
                            VALUES (?, ?, ?, ?)
                        """, [comment_id, topic_id, user_id, new_comment_text])

                        # Append new comment to history
                        st.session_state.comment_history += f"\n\n💬 {new_comment_text}"

                        # Get next comment (could be the one just submitted)
                        next_comment_id, next_comment_content = get_random_unvoted_comment(user_id, topic_id)
                        st.session_state.current_comment_id = next_comment_id
                        st.session_state.current_comment_content = next_comment_content

                        # Update progress
                        update_user_progress(user_id, topic_id, next_comment_id)

                        st.session_state.new_comment_input = "" # Clear input box
                        st.rerun() # Rerun to update UI

                    except Exception as e:
                        st.error(f"Error sharing information: {e}")
                    finally:
                        if local_con:
                            local_con.close()
                else:
                    st.error("Could not find or create user.")
            else:
                st.warning("Please enter your thought.")

    st.markdown("---")


    if st.button("Pack all insights and Return to Capital"):
        st.session_state.page = 'home'
        st.rerun()

    # st.components.v1.html("""
    #     <script>
    #     document.addEventListener('DOMContentLoaded', function() {
    #         // 监听 DOM 变化
    #         const observer = new MutationObserver(() => scrollToTarget());
    #         observer.observe(document.body, {
    #             childList: true,
    #             subtree: true,
    #             attributes: true
    #         });
    #     });


    #     // 滚动到目标元素
    #     function scrollToTarget() {
    #         const target = document.querySelector("div.stColumn");
    #         if (target) {
    #             target.scrollIntoView({ behavior: "smooth", block: "center" });
    #         }
    #     }
    #     </script>""")

# Initialize session state for navigation and data
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_topic_id' not in st.session_state:
    st.session_state.current_topic_id = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = '' # Mimics browser state
if 'current_comment_id' not in st.session_state:
    st.session_state.current_comment_id = None
if 'current_comment_content' not in st.session_state:
    st.session_state.current_comment_content = "Loading comments..."
if 'comment_history' not in st.session_state:
    st.session_state.comment_history = ""
if 'processed_url_params' not in st.session_state:
    st.session_state.processed_url_params = False # Add flag initialization

# Initialize the database and add dummy data only once per session
if st.session_state.get("db_initialized", False) is False:
    print("INFO: Initializing database and adding dummy data...") # Optional: Info message
    initialize_database()
    add_dummy_topic()
    st.session_state.db_initialized = True
    print("INFO: Database initialization complete.") # Optional: Info message


# Handle initial load from URL query parameters
# Process only once per session load using the flag
query_params = st.query_params
# Check for 'topic' param and if it hasn't been processed yet
if 'topic' in query_params and not st.session_state.processed_url_params:
    topic_id_from_url = query_params.get('topic') # Use .get for safety
    if topic_id_from_url: # Check if topic_id is actually retrieved
        st.session_state.page = 'view_topic'
        st.session_state.current_topic_id = topic_id_from_url
        st.session_state.processed_url_params = True # Mark as processed
        # The view_topic_page will handle loading user/comment based on session_state.user_email
        st.rerun() # Rerun to apply the page change


# Render the appropriate page based on session state
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'create_topic':
    create_topic_page()
elif st.session_state.page == 'view_topic':
    view_topic_page()
