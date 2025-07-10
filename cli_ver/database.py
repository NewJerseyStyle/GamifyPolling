# database.py
import duckdb
import uuid
import random
import os

# Database file path
DB_PATH = 'polis_archive.duckdb'

def initialize_database():
    """Initializes the DuckDB database and creates tables if they don't exist."""
    is_new_db = not os.path.exists(DB_PATH)
    con = duckdb.connect(database=DB_PATH, read_only=False)
    try:
        # --- Table Definitions (Unchanged) ---
        con.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("""
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
        con.execute("""
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
        con.execute("""
            CREATE TABLE IF NOT EXISTS seen_comments (
                user_id TEXT NOT NULL,
                comment_id TEXT NOT NULL,
                seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, comment_id)
            )
        """)

        con.execute("INSERT INTO users (id, username) VALUES ('system', 'System') ON CONFLICT (username) DO NOTHING")

        # If the database was just created, populate it with the example.
        if is_new_db:
            print("New database created. Adding fully populated example Inquiry...")
            add_example_topic()

    finally:
        con.close()

# REWRITTEN to include random votes
def add_example_topic():
    """
    Adds a pre-populated example topic for demonstration, including
    seed comments, dummy users, and random votes to simulate an active Inquiry.
    """
    topic_id = "main-street-2025"
    topic_title = "Inquiry #2025-08: The Main Street Redevelopment"
    topic_description = "Document the diverse public perspectives on the 'Main Street Redevelopment' of 2025, a key event in our city's history. The testimonies you collect and the insights you provide will form the basis of our historical understanding."
    comments_list = [
        "Turning Main Street into a pedestrian-only zone will kill local businesses. Where will people park?",
        "I'm tired of dodging cars on my bike. Fully protected bike lanes are a matter of public safety.",
        "This whole redevelopment is just a handout to big developers. What about preserving the historic character of our town?",
        "More green space and outdoor seating would make downtown a place people actually want to spend time in.",
        "Just fix the potholes. We don't need a grand, expensive plan; we need basic maintenance.",
        "A vibrant, walkable downtown will increase property values for everyone. It's a long-term investment.",
        "I run a delivery service. If you close Main Street to traffic, my job becomes impossible.",
        "Think of the elderly and people with disabilities. A car-free street is not accessible for everyone.",
        "This is our chance to create a modern, climate-friendly city center. We should be bold!",
        "What about the impact on property taxes? Will this redevelopment make our city unaffordable for long-time residents?",
        "I'd love to see more public art installations and community gathering spaces. That's what truly builds a sense of place.",
        "We need better public transit options *before* we restrict car access. Don't put the cart before the horse.",
        "This is a fantastic opportunity to attract new businesses and tourists. We need to modernize to stay competitive.",
        "My concern is the timeline. How long will Main Street be disrupted during construction? Small businesses can't afford long closures.",
        "Let's ensure there's a strong focus on local, independent businesses, not just chain stores. Support our community!",
        "Will there be enough public restrooms and waste disposal facilities if more people are spending time downtown?",
        "I hope the new design considers shade and cooling elements for our hot summers. Green infrastructure is key.",
        "This plan feels like it's designed for young, affluent people. What about families and those on a budget?",
        "We need clear communication and transparency throughout this entire process. Don't just tell us, involve us!",
        "Consider the noise pollution during construction. Residents living nearby deserve peace.",
        "Will there be designated areas for street performers and vendors? That adds so much character!",
        "Let's not forget about safety. Good lighting and visible security are crucial for a thriving public space.",
        "I'm excited about the potential for more community events and festivals with a revitalized Main Street.",
        "How will this affect emergency vehicle access? That's a critical consideration for any street closure.",
    ]
    
    with duckdb.connect(database=DB_PATH, read_only=False) as con:
        # --- Step 1: Create the Topic and Seed Comments ---
        con.execute("INSERT INTO topics (id, name, description) VALUES (?, ?, ?) ON CONFLICT (id) DO NOTHING", [topic_id, topic_title, topic_description])
        
        comment_ids = []
        for comment_text in comments_list:
            comment_id = str(uuid.uuid4())
            con.execute("INSERT INTO comments (id, topic_id, user_id, content) VALUES (?, ?, ?, ?)", [comment_id, topic_id, 'system', comment_text])
            comment_ids.append(comment_id)
        
        print(f"Successfully added example Inquiry: '{topic_title}' (ID: {topic_id}) with {len(comment_ids)} seed comments.")

        # --- Step 2: Create Dummy Users ---
        num_dummy_users = 20
        dummy_user_ids = []
        for i in range(num_dummy_users):
            user_id = str(uuid.uuid4())
            # Use a unique, random username
            username = f"agent_{uuid.uuid4().hex[:6]}@time.gov"
            con.execute("INSERT INTO users (id, username) VALUES (?, ?) ON CONFLICT (username) DO NOTHING", [user_id, username])
            dummy_user_ids.append(user_id)
        
        print(f"Created {len(dummy_user_ids)} dummy users for seeding votes.")

        # --- Step 3: Generate and Insert Random Votes ---
        votes_to_insert = []
        for user_id in dummy_user_ids:
            for comment_id in comment_ids:
                # Give each user an 80% chance of voting on any given comment
                if random.random() < 0.8:
                    vote_id = str(uuid.uuid4())
                    # Skew votes slightly towards agree/disagree over neutral
                    vote_type = random.choice(['agree', 'agree', 'agree', 'disagree', 'disagree', 'neutral'])
                    votes_to_insert.append((vote_id, user_id, comment_id, vote_type))
        
        if votes_to_insert:
            con.executemany("""
                INSERT INTO votes (id, user_id, comment_id, vote_type) VALUES (?, ?, ?, ?)
            """, votes_to_insert)
        
        print(f"Added {len(votes_to_insert)} random votes to the example Inquiry.")

# NEW: Function to create a quest/topic from the UI
def create_new_topic(topic_id: str, title: str, description: str, seed_comments: list[str]):
    """Creates a new topic and populates it with initial seed comments."""
    with duckdb.connect(database=DB_PATH, read_only=False) as con:
        # Insert the topic
        con.execute("INSERT INTO topics (id, name, description) VALUES (?, ?, ?) ON CONFLICT (id) DO NOTHING", [topic_id, title, description])

        # Insert comments from the 'system' user
        for comment_text in seed_comments:
            if comment_text.strip(): # Ensure we don't add empty lines
                comment_id = str(uuid.uuid4())
                con.execute("INSERT INTO comments (id, topic_id, user_id, content) VALUES (?, ?, ?, ?)", [comment_id, topic_id, 'system', comment_text])
        print(f"Successfully created Inquiry: '{title}' (ID: {topic_id})")

def find_or_create_user(username: str) -> str:
    """Finds a user by username or creates a new one, returning the user ID."""
    with duckdb.connect(database=DB_PATH, read_only=False) as con:
        user_result = con.execute("SELECT id FROM users WHERE username = ?", [username]).fetchone()
        if user_result:
            return user_result[0]
        else:
            user_id = str(uuid.uuid4())
            con.execute("INSERT INTO users (id, username) VALUES (?, ?)", [user_id, username])
            return user_id

def get_topic_details(topic_id: str) -> tuple | None:
    """Fetches the name and description for a given topic ID."""
    with duckdb.connect(database=DB_PATH, read_only=True) as con:
        return con.execute("SELECT name, description FROM topics WHERE id = ?", [topic_id]).fetchone()

def record_vote(user_id: str, comment_id: str, vote_type: str):
    """Records or updates a user's vote on a comment."""
    with duckdb.connect(database=DB_PATH, read_only=False) as con:
        vote_id = str(uuid.uuid4())
        con.execute("""
            INSERT INTO votes (id, user_id, comment_id, vote_type) VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id, comment_id) DO UPDATE SET vote_type = EXCLUDED.vote_type
        """, [vote_id, user_id, comment_id, vote_type])

def add_new_comment(user_id: str, topic_id: str, content: str) -> str:
    """Adds a new comment to a topic from a user."""
    with duckdb.connect(database=DB_PATH, read_only=False) as con:
        comment_id = str(uuid.uuid4())
        con.execute("INSERT INTO comments (id, topic_id, user_id, content) VALUES (?, ?, ?, ?)",
                      [comment_id, topic_id, user_id, content])
        return comment_id

def mark_comment_as_seen(user_id: str, comment_id: str):
    """Marks a comment as seen by a user."""
    with duckdb.connect(database=DB_PATH, read_only=False) as con:
        con.execute("INSERT INTO seen_comments (user_id, comment_id) VALUES (?, ?) ON CONFLICT DO NOTHING", [user_id, comment_id])
