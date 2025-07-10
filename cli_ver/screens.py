# screens.py
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Input, RichLog, Static, Label, TextArea
from textual.containers import Vertical, VerticalScroll, Horizontal
import random
import uuid

import database
import game_engine

# MODIFIED: New prompts for variety
NARRATIVE_PROMPTS = [
    "You uncover a new testimony from the archive...",
    "A voice from the past echoes through the data stream...",
    "Another perspective materializes from the records...",
    "The archive yields another fragment of public opinion...",
    "A citizen's report from 2025 appears on your console...",
    "Sifting through the noise, you isolate a distinct viewpoint...",
]

class TitleScreen(Screen):
    """The first screen the user sees. For login or creating a new inquiry."""
    CSS_PATH = "app.css"

    def compose(self) -> ComposeResult:
        yield Header(name="Temporal Correspondence Archive")
        with VerticalScroll(id="title_form"):
            yield Label("Join an existing Inquiry:")
            yield Input(placeholder="Your Email (e.g., agent@time.gov)", id="email")
            yield Input(placeholder="Inquiry ID (e.g., main-street-2025)", id="topic")
            yield Button("Begin Inquiry", variant="primary", id="start")
            yield Label("- or -")
            yield Button("Create New Inquiry", variant="success", id="create_new")
            # NEW: Button to create a quest
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            email = self.query_one("#email", Input).value
            topic_id = self.query_one("#topic", Input).value
            if email and topic_id:
                topic_details = database.get_topic_details(topic_id)
                if topic_details:
                    user_id = database.find_or_create_user(email)
                    self.app.push_screen(GameScreen(user_id=user_id, topic_id=topic_id))
                else:
                    self.query_one("#topic", Input).value = ""
                    self.query_one("#topic", Input).placeholder = "INVALID INQUIRY ID"
            else:
                self.notify("All fields are required.", severity="error", timeout=5)
        
        # NEW: Handle navigation to the create screen
        elif event.button.id == "create_new":
            self.app.push_screen(CreateInquiryScreen())


# NEW: Screen for creating a quest
class CreateInquiryScreen(Screen):
    """A screen for administrators to create a new Inquiry."""
    CSS_PATH = "app.css"

    def compose(self) -> ComposeResult:
        yield Header(name="Archive Administration - Create New Inquiry")
        with VerticalScroll(id="create_form"):
            yield Label("Inquiry ID (short, no spaces, e.g., 'park-funding-2025'):")
            yield Input(placeholder="Inquiry ID", id="new_topic_id")
            
            yield Label("\nInquiry Title (The full question or name):")
            yield Input(placeholder="Title", id="new_topic_title")

            yield Label("\nMission Briefing (A description for the Correspondents):")
            yield TextArea("", id="new_topic_desc")
            yield Static("[italic]Player will see this to understand their role and the information we are expecting to collect.[/italic]")

            yield Label("\nSeed Testimonies (One opinion per line to start the archive):")
            yield TextArea("", id="seed_comments")
            yield Static("[italic]Good Seed Testimonies make people think what do they want and who they are (Some opinions most likely to have some people agree while some others will disagree)[/italic]\n")

            with Horizontal():
                yield Button("Create Inquiry in Archive", variant="success", id="submit_create")
                yield Button("Cancel", variant="error", id="cancel_create")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel_create":
            self.app.pop_screen()
        elif event.button.id == "submit_create":
            topic_id = self.query_one("#new_topic_id", Input).value.strip()
            title = self.query_one("#new_topic_title", Input).value.strip()
            desc = self.query_one("#new_topic_desc", TextArea).text.strip()
            # The text attribute of TextArea includes newlines
            seed_comments = self.query_one("#seed_comments", TextArea).text.split('\n')

            if topic_id and title and desc and any(seed_comments):
                # We use the user-provided ID. uuid could be a fallback.
                database.create_new_topic(topic_id, title, desc, seed_comments)
                self.app.pop_screen()
            else:
                self.query_one(Footer).notify("All fields are required.", severity="error", timeout=5)


# REWRITTEN: The GameScreen is now command-driven
class GameScreen(Screen[None]):
    """The main game screen where the Zork-style interaction happens."""
    CSS_PATH = "app.css"

    def __init__(self, user_id: str, topic_id: str):
        super().__init__()
        self.user_id = user_id
        self.topic_id = topic_id
        self.topic_name, self.topic_description = database.get_topic_details(topic_id)
        
        # Game State
        self.current_comment_id = None
        self.current_comment_content = ""
        self.current_cluster_label = None
        self.current_town_name = "The Chronos Anomaly"
        self.testimonies_collected = 0

    def compose(self) -> ComposeResult:
        yield Header(name=self.topic_name)
        with Vertical(id="game_screen_container"):
            yield RichLog(highlight=True, markup=True, wrap=True, id="narrative_log")
            yield Input(placeholder="Type commands here (e.g., 'agree', 'help')", id="command_input")
        yield Footer()

    async def on_mount(self) -> None:
        """Called when the screen is first mounted."""
        log = self.query_one(RichLog)
        log.write(f"[bold green]Connection established.[/bold green] Welcome, Temporal Correspondent.")
        log.write(f"You are now accessing the historical record for: [bold cyan]{self.topic_name}[/bold cyan]")
        log.write(f"[italic]Mission Briefing: {self.topic_description}[/italic]")
        log.write("---")
        self.query_one(Input).focus()
        
        await self.update_town_and_location()
        await self.fetch_and_display_next_opinion()

    async def update_town_and_location(self):
        """Checks the user's cluster and updates the town name if it has changed."""
        log = self.query_one(RichLog)
        new_label, cluster_users = game_engine.get_user_cluster_info(self.user_id, self.topic_id)

        if self.current_cluster_label is None:
            self.query_one(Footer).notify(f"Current Location: {self.current_town_name}")

        if new_label is not None and new_label != self.current_cluster_label:
            self.current_cluster_label = new_label
            old_town_name = self.current_town_name
            self.current_town_name, desc = game_engine.name_user_group(cluster_users, self.topic_id)
            if old_town_name != self.current_town_name:
                log.write(f"[bold yellow]Your perspective has shifted. You have traveled from {old_town_name} to...[/bold yellow]")
                log.write(f"[bold magenta]Location: {self.current_town_name}[/bold magenta]\n[italic]{desc}[/italic]")
                self.query_one(Footer).notify(f"Current Location: {self.current_town_name}")
        return cluster_users

    async def fetch_and_display_next_opinion(self):
        """Gets the next opinion and updates the UI."""
        log = self.query_one(RichLog)
        cluster_users = game_engine.get_user_cluster_info(self.user_id, self.topic_id)[1]

        comment_id, content, is_repeat = game_engine.get_next_opinion(self.user_id, self.topic_id, cluster_users)
        
        self.current_comment_id = comment_id
        self.current_comment_content = content

        if comment_id:
            database.mark_comment_as_seen(self.user_id, comment_id)
            if is_repeat:
                log.write(f"\n[italic]A familiar sentiment echoes in {self.current_town_name}...[/italic]")
            else:
                log.write(f"\n[italic]{random.choice(NARRATIVE_PROMPTS)}[/italic]")
            
            # Display the opinion in a formatted block
            log.write(f"[bold]{self.current_comment_content}[/bold]")
            log.write("--------------------")

        else:
            log.write(f"\n[bold red]{content}[/bold red]")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user commands."""
        command_text = event.value.strip()
        log = self.query_one(RichLog)
        self.query_one("#command_input", Input).value = "" # Clear input

        log.write(f"> [dim]{command_text}[/dim]") # Echo command

        parts = command_text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # --- Command Parsing ---
        if command in ("agree", "a", "corroborate"):
            if self.current_comment_id:
                database.record_vote(self.user_id, self.current_comment_id, "agree")
                self.testimonies_collected += 1
                log.write("Your corroboration has been archived. Analyzing next testimony...")
                await self.update_town_and_location()
                await self.fetch_and_display_next_opinion()
            else:
                log.write("[red]There is no active testimony to agree with.[/red]")

        elif command in ("disagree", "d", "contest"):
            if self.current_comment_id:
                database.record_vote(self.user_id, self.current_comment_id, "disagree")
                self.testimonies_collected += 1
                log.write("Your contestation has been archived. Analyzing next testimony...")
                await self.update_town_and_location()
                await self.fetch_and_display_next_opinion()
            else:
                log.write("[red]There is no active testimony to disagree with.[/red]")

        elif command in ("submit", "report", "say"):
            if args:
                database.add_new_comment(self.user_id, self.topic_id, args)
                log.write("[green]Your report has been successfully filed in the 2025 archive.[/green]")
            else:
                log.write("[yellow]Usage: submit <your full testimony>[/yellow]")

        elif command in ("look", "l", "read"):
            log.write(f"\n[italic]Rereading the current testimony from {self.current_town_name}:[/italic]")
            log.write(f"[bold]{self.current_comment_content}[/bold]")
        
        elif command in ("help", "h", "?"):
            log.write("\n[bold yellow]--- Temporal Correspondent Commands ---[/bold yellow]")
            log.write("[bold]agree[/bold] or [bold]a[/bold] - Corroborate the current testimony.")
            log.write("[bold]disagree[/bold] or [bold]d[/bold] - Contest the current testimony.")
            log.write("[bold]submit <text>[/bold] - Submit your own testimony to the archive.")
            log.write("[bold]look[/bold] or [bold]l[/bold] - Reread the current testimony.")
            log.write("[bold]whereami[/bold] - Check your current location in the opinion space.")
            log.write("[bold]quit[/bold] or [bold]q[/bold] - Exit the archive and end your session.")

        elif command in ("whereami", "location"):
            log.write(f"You are currently in [bold magenta]{self.current_town_name}[/bold magenta].")

        elif command in ("quit", "q", "exit"):
            log.write("[bold yellow]Ending session. The archive is grateful for your contributions.[/bold yellow]")
            self.app.exit()

        else:
            log.write(f"[red]Command '{command}' not understood. Type 'help' for a list of commands.[/red]")