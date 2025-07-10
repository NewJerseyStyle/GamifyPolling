# Textual ver.
### Pending Feature: Discord Integration

The last major piece from the original request is the **Discord achievement system**.

**Plan for Implementation:**

1.  **Library:** We will need to add the `discord-webhook` library.
    `pip install discord-webhook`
2.  **Configuration:** The app will need a place to store the Discord webhook URL, perhaps in a simple `config.ini` file or as an environment variable for security.
3.  **Logic (`game_engine.py`):** We'll create a function `check_and_grant_achievements(user_id, topic_id)`. This function will be called after every vote.
4.  **Triggers:** Achievements could be triggered by:
    *   `First Vote`: "First Step into the Past"
    *   `Submit 5 Testimonies`: "Prolific Chronicler"
    *   `Visit 3 Different Towns`: "Temporal Traveler"
    *   `Vote on a highly contested issue`: "Chronicler of Dissent"
5.  **Notification:** When an achievement is granted, it will `log.write` a message to the in-game console and call a `discord.py` function to send a nicely formatted embed message to the configured webhook.
