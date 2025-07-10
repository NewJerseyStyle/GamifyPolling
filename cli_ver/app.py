# app.py
from textual.app import App
from screens import TitleScreen, GameScreen, CreateInquiryScreen # MODIFIED
import database

class TemporalApp(App):
    """A Textual app for temporal correspondence."""

    # MODIFIED: Added create_inquiry screen
    SCREENS = {
        "title": TitleScreen,
        "game": GameScreen,
        "create_inquiry": CreateInquiryScreen
    }

    def on_mount(self) -> None:
        """Mount the title screen on startup."""
        self.push_screen("title")

if __name__ == "__main__":
    # Initialize the database on first run
    database.initialize_database()
    app = TemporalApp()
    app.run()