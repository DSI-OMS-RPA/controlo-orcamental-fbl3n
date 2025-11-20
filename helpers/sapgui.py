from pdb import run

import pandas as pd
from requests import session
from helpers.utils import setup_logger, run_application
from datetime import datetime, timedelta
from pywinauto.application import Application
from dateutil.relativedelta import relativedelta
from helpers.configuration import *
import win32com.client as win32
import win32clipboard as wcb
import pygetwindow as gw
from typing import Tuple
import win32gui
import win32con
import random
import locale
import time
import sys

# Initialize the logger manager
logger = setup_logger(__name__)

class SapGui():
    """
    Represents a SAP GUI session.

    Args:
        sap_args (dict): Dictionary containing SAP login credentials.

    Raises:
        Exception: If there is an error during initialization.

    Examples:
        sap_session = SapGui(sap_args)
    """

    def __init__(self, sap_args):
        """
        Initializes a SAP GUI session.

        Args:
            sap_args (dict): Dictionary containing SAP login credentials.

        Raises:
            Exception: If there is an error during initialization.
        """
        try:

            # Initialize instance variables for SAP configurations
            self.system = sap_args["platform"]
            self.client = sap_args['client']
            self.user = sap_args["username"]
            self.password = sap_args["password"]
            self.language = sap_args['language']
            self.path = sap_args['path']

            # Get timing parameters from config or use defaults
            self.max_wait_time = sap_args['max_wait_time']
            self.retry_interval = sap_args['retry_interval']

            # Initialize connection objects
            self.SapGuiAuto = None
            self.connection = None
            self.session = None

            # Step 1: Ensure SAP Logon is running
            if not self._ensure_sap_logon_running():
                raise Exception("Failed to start SAP Logon")

            # Step 2: Wait for SAP GUI to be ready with COM interface
            if not self._wait_for_sap_gui_ready(self.max_wait_time, self.retry_interval):
                raise Exception(f"SAP GUI not ready within {self.max_wait_time} seconds")

            # Step 3: Initialize SAP GUI objects
            self._initialize_sap_objects()

            logger.info("SAP GUI session initialized successfully")

        except Exception as e:
            logger.error(f"An exception occurred in __init__: {str(e)}")
            self._cleanup_on_error()
            raise

    def _ensure_sap_logon_running(self) -> bool:
        """
        Ensure SAP Logon is running, start it if necessary.

        Returns:
            bool: True if SAP Logon is running, False otherwise
        """
        try:
            # Check if already running
            if self._is_sap_logon_running():
                logger.info("SAP Logon is already running")
                return True

            # Try to start SAP Logon
            logger.info("Starting SAP Logon...")
            runner = run_application(self.path)
            if not runner:
                logger.error("Failed to start SAP Logon")
                return False

            # Wait a bit for the process to start
            time.sleep(3)

            # Verify it's running
            if self._is_sap_logon_running():
                logger.info("SAP Logon started successfully")
                return True
            else:
                logger.error("SAP Logon failed to start properly")
                return False

        except Exception as e:
            logger.error(f"Error ensuring SAP Logon is running: {str(e)}")
            return False

    def _is_sap_logon_running(self) -> bool:
        """
        Check if SAP Logon process is running.

        Returns:
            bool: True if running, False otherwise
        """
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name']):
                if 'saplogon' in proc.info['name'].lower():
                    return True
            return False
        except Exception:
            return False

    def _wait_for_sap_gui_ready(self, max_wait_time: int, retry_interval: int) -> bool:
        """
        Wait for SAP GUI to be ready for COM connections.

        Args:
            max_wait_time (int): Maximum time to wait
            retry_interval (int): Interval between checks

        Returns:
            bool: True if SAP GUI is ready, False if timeout
        """
        start_time = time.time()
        attempts = 0

        logger.info(f"Waiting for SAP GUI to be ready (max {max_wait_time}s)...")

        while time.time() - start_time < max_wait_time:
            attempts += 1
            try:
                # Try to get the SAP GUI object
                sap_gui = win32.GetObject("SAPGUI")
                if sap_gui and isinstance(sap_gui, win32.CDispatch):
                    # Try to get the scripting engine
                    scripting_engine = sap_gui.GetScriptingEngine
                    if scripting_engine and isinstance(scripting_engine, win32.CDispatch):
                        logger.info(f"SAP GUI ready after {attempts} attempts ({time.time() - start_time:.1f}s)")
                        return True

            except Exception as e:
                logger.debug(f"Attempt {attempts}: SAP GUI not ready yet - {str(e)}")

            time.sleep(retry_interval)

        logger.error(f"SAP GUI not ready after {max_wait_time}s and {attempts} attempts")
        return False

    def _initialize_sap_objects(self):
        """
        Initialize SAP GUI COM objects.

        Raises:
            Exception: If any SAP object cannot be initialized
        """
        try:
            # Get SAP GUI object
            self.SapGuiAuto = win32.GetObject("SAPGUI")
            if not isinstance(self.SapGuiAuto, win32.CDispatch):
                raise Exception("Failed to get SAP GUI object")

            # Get scripting engine
            application = self.SapGuiAuto.GetScriptingEngine
            if not isinstance(application, win32.CDispatch):
                raise Exception("Failed to get SAP scripting engine")

            # Open connection
            logger.info(f"Opening connection to SAP system: {self.system}")
            self.connection = application.OpenConnection(self.system, True)
            if not isinstance(self.connection, win32.CDispatch):
                raise Exception("Failed to open SAP connection")

            # Wait for connection to be established
            time.sleep(3)

            # Get session
            self.session = self.connection.Children(0)
            if not isinstance(self.session, win32.CDispatch):
                raise Exception("Failed to get SAP session")

            # Resize window
            try:
                self.session.findById("wnd[0]").resizeWorkingPane(169, 30, False)
            except Exception as e:
                logger.warning(f"Failed to resize window: {str(e)}")

            logger.info("SAP objects initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SAP objects: {str(e)}")
            raise

    def _cleanup_on_error(self):
        """
        Clean up resources when initialization fails.
        """
        try:
            if hasattr(self, 'session') and self.session:
                self.session = None
            if hasattr(self, 'connection') and self.connection:
                self.connection = None
            if hasattr(self, 'SapGuiAuto') and self.SapGuiAuto:
                self.SapGuiAuto = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def is_connected(self) -> bool:
        """
        Check if SAP GUI session is properly connected.

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            return (self.SapGuiAuto is not None and
                    self.connection is not None and
                    self.session is not None)
        except Exception:
            return False

    def handle_password_change(self):
        """
        Handles the password change prompt during SAP login.

        This function detects a password change popup, generates a new password in the format
        'Month#Year' (e.g., 'Maio#2024'), inputs the password in both fields, and attempts to
        log the user in. If successful, it returns True; otherwise, it returns False.

        Returns:
            bool: True if the password change was successful and the user is logged in, False otherwise.
        """
        try:
            # Ensure the locale is set to Portuguese for correct month formatting
            try:
                locale.setlocale(locale.LC_TIME, 'pt_PT.UTF-8')
            except locale.Error as e:
                logger.error(f"Locale setting failed: {str(e)}")
                return False

            # Check if a password change window is active (usually wnd[1])
            active_window = self.session.ActiveWindow
            if active_window.Name != "wnd[1]":
                logger.info("No password change prompt detected.")
                return True

            # Retrieve the popup window and its title
            popup_window = self.session.findById("wnd[1]")
            popup_title = popup_window.Text.lower()

            # Ensure it's a password change prompt by checking the title
            if "sap" not in popup_title:
                logger.error("Unexpected popup encountered, not a password change prompt.")
                return False

            # Verify the label to confirm it's asking for a new password
            try:
                input_label = popup_window.findById("usr/lblRSYST-NCODE_TEXT").Text.lower()
                if "nova senha" not in input_label:
                    logger.error("No valid password change prompt detected.")
                    return False
            except Exception as e:
                logger.error(f"Failed to find password change label: {str(e)}")
                return False

            logger.info("Password change prompt detected.")

            # Generate new password in 'NumberMonth#Year' format (e.g., '123Maio#2024')
            number = random.randrange(1, 999)

            new_password = f"{number}{datetime.now().strftime('%B').capitalize()}#{datetime.now().strftime('%Y')}"
            logger.info(f"Generated new password: {new_password}")

            # Input the new password into both fields
            popup_window.findById("usr/pwdRSYST-NCODE").text = new_password
            popup_window.findById("usr/pwdRSYST-NCOD2").text = new_password

            # Confirm the password change
            popup_window.findById("tbar[0]/btn[0]").press()
            logger.info(f"Password change confirmed:{new_password}")

            # Wait for a short while to allow the change to take effect
            time.sleep(3)

            # Check if the user is logged in successfully after password change
            if self.session.findById("wnd[0]/tbar[0]/btn[15]", False):
                logger.info("Password changed successfully and user logged in.")
                return True
            else:
                logger.error("Password change failed or login unsuccessful.")
                return False

        except Exception as e:
            # Log any errors that occur during the process
            logger.error(f"Error during password change handling: {str(e)}")
            return False

    # Method to login to SAP using the SAP GUI Scripting API and the win32 library.
    def sapLogin(self):
        """
        Logs in to SAP.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        try:

            # Verify we have a valid session
            if not self.is_connected():
                logger.error("SAP session not properly initialized")
                return False

            logger.info("Attempting SAP login...")

            # Set the SAP login credentials and language in the GUI
            self.session.findById("wnd[0]/usr/txtRSYST-MANDT").text = self.client  # Mandante
            self.session.findById("wnd[0]/usr/txtRSYST-BNAME").text = self.user  # Utilizador
            self.session.findById("wnd[0]/usr/pwdRSYST-BCODE").text = self.password  # Password
            self.session.findById("wnd[0]/usr/txtRSYST-LANGU").text = self.language  # Idioma

            # Perform the login
            self.session.findById("wnd[0]").sendVKey(0)

            # Wait for a short time to see if any popup appears
            time.sleep(2)

            # Check for the specific popup window
            if self.session.ActiveWindow.Name == "wnd[1]":
                popup_text = self.session.findById("wnd[1]").Text
                logger.info(f"Popup detected: {popup_text}")

                # Check if the popup is the multiple login warning by checking some unique text or title
                if "logon múltiplo" in popup_text:
                    logger.info("Multiple logins detected. Closing other sessions.")
                    # Select the option to close other sessions and click OK
                    self.session.findById("wnd[1]/usr/radMULTI_LOGON_OPT1").select()
                    self.session.findById("wnd[1]/tbar[0]/btn[0]").press()

                    # click enter to close the popup
                    shell = win32.Dispatch("WScript.Shell")
                    shell.SendKeys("{ENTER}", 0)

            # Handle password change if required
            if not self.handle_password_change():
                # If password change handling fails, return False
                return False

            # Check if login is successful by finding a UI element that appears only when logged in
            if self.session.findById("wnd[0]/tbar[0]/btn[15]"):
                # login sucessfull
                logger.info("Successfully connected to SAP.")
                return True
            else:
                # Login failed and Close the SAP GUI connection
                self.close_connection()
                return False

        except Exception as e:
            logger.error(f"Error during SAP login: {str(e)}")
            logger.error(sys.exc_info())

    # A method to close a SAP connection
    def close_connection(self):
        """
        Close SAP connection with improved error handling.
        """
        # Check if a connection object exists
        try:
            if self.connection is not None:
                self.connection.CloseSession('ses[0]')
                # Set the connection to None, indicating it's closed
                self.connection = None
                # Log a message indicating that the SAP connection is closed
                logger.info("SAP connection closed.")
            if self.SapGuiAuto is not None:
                self.SapGuiAuto = None
            logger.info("SAP connection closed safely.")
        except Exception as e:
            # Handle any exceptions that may occur during the closing process
            # Log an error message with details about the exception
            logger.error(f"Error closing SAP connection: {str(e)}")


    def sapLogout(self):
        """
        Logs out of SAP.

        Args:
            self: The instance of the SAP session.

        Raises:
            Exception: If there is an error during SAP logout.

        Examples:
            sap_session = SAPSession()
            sap_session.sapLogout()
        """
        try:

            if not self.is_connected():
                logger.warning("Not connected to SAP, cannot logout")
                return

            # Enter the logout command '/nex' in the command field
            self.session.findById("wnd[0]/tbar[0]/okcd").text = "/nex"
            self.session.findById("wnd[0]").sendVKey(0)

            logger.info("Successfully logged out of SAP.")
        except Exception as e:
            logger.error(f"Error during SAP logout: {str(e)}")


    @staticmethod
    def get_dates():
        current_date = datetime.now()

        # Get the first day of the current month
        first_day_current_month = current_date.replace(day=1)

        # Subtract one day from the first day of the current month to get the last day of the previous month
        previous_month_last_day = first_day_current_month - timedelta(days=1)

        # Set the start_date as the first day of the previous month
        start_date = previous_month_last_day.replace(day=1).strftime("%d.%m.%Y")
        end_date = current_date.strftime("%d.%m.%Y")

        return start_date, end_date


    # Waits for the SAP GUI element with the specified ID to appear within the given timeout.
    def wait_for_element(self, element_id, timeout=60):
        """
        Args:
            element_id (str): SAP GUI Scripting ID of the element to wait for.
            timeout (int, optional): The number of seconds to wait for the element. Defaults to 60.

        Returns:
            bool: True if the element appears within the timeout, otherwise False.
        """
        if not self.is_connected():
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.session.findById(element_id):
                    return True
            except Exception:
                time.sleep(1)  # wait for 1 second before trying again
        return False

    def check_element_exists(self, element_path):
        """
        Checks if a SAP element exists.

        Args:
            element_path (str): The path of the SAP element.

        Returns:
            bool: True if the element exists, otherwise False.
        """

        try:
            self.session.findById(element_path)
            return True
        except Exception:
            return False

    def wait_for_save_as_dialog(self, title):
        """
        Waits for a dialog window to appear.
        This function waits for a maximum number of attempts for a dialog window to appear. It checks if the window is present and returns True if it is, otherwise it returns False.
        """
        # Maximum number of attempts to wait for the Save As dialog
        max_attempts = 10

        for _ in range(max_attempts):
            if gw.getWindowsWithTitle(title):
                return True
            else:
                time.sleep(1)  # Wait for 1 second before the next attempt

        return False

    def get_sap_element_text(self, element_path):
        """
        Retrieves the text of a SAP element identified by the given element_path.

        Args:
            element_path (str): The path of the SAP element.

        Returns:
            str: The text of the SAP element, or None if the element is not found or an error occurs.
        """
        try:
            element = self.session.FindById(element_path)
            return element.Text
        except Exception as e:
            print(f"Error: {e}")
            return None

    def bring_dialog_to_top(self, title):
        """
        Brings a dialog window to the top of the screen.

        This function checks if a dialog window is open and, if so, restores and shows the window, and brings it to the top of the screen.
        """
        save_as_window = gw.getWindowsWithTitle(title)
        if save_as_window:
            # Restore the window if minimized
            win32gui.ShowWindow(save_as_window[0]._hWnd, win32con.SW_RESTORE)
            win32gui.ShowWindow(save_as_window[0]._hWnd, win32con.SW_SHOWNORMAL)  # Show the window
            win32gui.BringWindowToTop(save_as_window[0]._hWnd)  # Bring the window to the top

    def scroll_to_field(self, field_path):
        try:
            self.session.findById(field_path).setFocus()
        except Exception:
            self.session.findById(field_path.split("/")[:-1].join("/")).verticalScrollbar.position += 1

    def close_sap_security_popup(self, delay: int = 2, max_wait: int = 10) -> bool:
        """
        Detects and clicks the 'Permitir' button on the SAP GUI security popup using pywinauto.

        Args:
            delay (int): Seconds to wait before attempting detection (default is 2).
            max_wait (int): Total seconds to wait before giving up (default is 10).

        Returns:
            bool: True if the popup was found and closed; False otherwise.
        """
        from pywinauto.application import Application
        import pywinauto
        import time

        try:
            logger.info("A verificar se existe popup de segurança SAPGUI...")

            # Wait a bit to let popup appear
            time.sleep(delay)

            app = Application(backend="uia")

            # Try repeatedly for a few seconds
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    # List all windows with similar title
                    windows = pywinauto.findwindows.find_windows(title_re=".*Segurança SAPGUI.*")
                    if windows:
                        dlg = app.window(handle=windows[0])
                        dlg.set_focus()
                        dlg.child_window(title="Permitir", control_type="Button").click_input()
                        logger.info("Popup de segurança SAPGUI detectado e botão 'Permitir' clicado com sucesso.")
                        return True
                except Exception:
                    time.sleep(1)

            logger.info("Nenhum popup de segurança SAPGUI foi detectado.")
            return False

        except Exception as e:
            logger.warning(f"Erro ao tentar fechar popup de segurança SAPGUI: {e}")
            return False

    def set_clipboard(self, text: str):
        """
        Sets the system clipboard to the specified text.
        Args:
            text (str): The text to set in the clipboard.
        """
        wcb.OpenClipboard()
        try:
            wcb.EmptyClipboard()
            wcb.SetClipboardText(text)
        finally:
            wcb.CloseClipboard()

    def sapOperation(self, command: str, client_accounts: list) -> Tuple[bool, str]:
        """
        Performs the specified command in the SAP GUI.

        Args:
            command (str): The command to be executed.

        Returns:
            None
        """
        try:

            if not self.is_connected():
                logger.error("Not connected to SAP")
                return False, None

            # Set the value of the specified field and Submit the command
            self.session.findById("wnd[0]/tbar[0]/okcd").text = command
            self.session.findById("wnd[0]").sendVKey(0)

            # Today (e.g., 2025-07-01)
            today = datetime.today()
            start_date = today.replace(day=1) - relativedelta(months=1) # First day of the previous month
            end_date = today.replace(day=1) - relativedelta(days=1) # Last day of the previous month

            # convert the start and end dates to the required format
            start_date = start_date.strftime("%d.%m.%Y")
            end_date = end_date.strftime("%d.%m.%Y")
            runtime = datetime.now().strftime("%Y%m%d%H%M%S")
            logger.info(f"Start date: {start_date} and End date: {end_date}")

            # Wait for element to be present
            element_to_wait_for = "wnd[0]/tbar[1]/btn[8]"
            if self.wait_for_element(element_to_wait_for):

                # Prepare client accounts list to copy and paste
                client_accounts_str = "\r\n".join(client_accounts)
                self.set_clipboard(client_accounts_str)

                # insert "Conta do cliente"
                self.session.findById("wnd[0]/usr/btn%_SD_SAKNR_%_APP_%-VALU_PUSH").press()
                self.session.findById("wnd[1]/tbar[0]/btn[24]").press() # Click past button
                self.session.findById("wnd[1]/tbar[0]/btn[8]").press() # Press Ok button

                # Clean input field "Aberto á data fixada"
                self.session.findById("wnd[0]/usr/ctxtPA_STIDA").text = ""

                # Select "Todas as partidas"
                self.session.findById("wnd[0]/usr/radX_AISEL").select()

                # insert "Data Lançamento"
                self.session.findById("wnd[0]/usr/ctxtSO_BUDAT-LOW").text = start_date
                self.session.findById("wnd[0]/usr/ctxtSO_BUDAT-HIGH").text = end_date

                # Choose "Layout"
                self.session.findById("wnd[0]/usr/ctxtPA_VARI").text = "/EXTRACTO_BI"

                # Press execute button
                self.session.findById("wnd[0]/tbar[1]/btn[8]").press()

                # Wait 3 seconds for the template to load
                time.sleep(3)

                # Wait for element to be present
                element_to_wait_for = "wnd[0]/tbar[1]/btn[43]"
                if self.wait_for_element(element_to_wait_for):

                    # Choose menu to export data - "Lista > Exportar > Filelocal"
                    self.session.findById("wnd[0]/mbar/menu[0]/menu[3]/menu[2]").select()

                    # Wait for element to be present
                    element_to_wait_for = "wnd[1]/usr/subSUBSCREEN_STEPLOOP:SAPLSPO5:0150/sub:SAPLSPO5:0150/radSPOPLI-SELFLAG[1,0]"
                    if self.wait_for_element(element_to_wait_for):
                        # Choose data format and type - "Planilha eletrônica"
                        self.session.findById("wnd[1]/usr/subSUBSCREEN_STEPLOOP:SAPLSPO5:0150/sub:SAPLSPO5:0150/radSPOPLI-SELFLAG[1,0]").select()
                        self.session.findById("wnd[1]/usr/subSUBSCREEN_STEPLOOP:SAPLSPO5:0150/sub:SAPLSPO5:0150/radSPOPLI-SELFLAG[1,0]").setFocus()
                        self.session.findById("wnd[1]/tbar[0]/btn[0]").press()

                    # Wait for element to be present
                    element_to_wait_for = "wnd[1]/usr/ctxtDY_PATH"
                    if self.wait_for_element(element_to_wait_for):

                        # Generate the filename and path
                        filename = f"export_partidas_aberto_{runtime}.txt"

                        # Set the file path based on project directory
                        project_root = Path(__file__).parent.parent
                        exports_folder = project_root / "resources"
                        exports_folder.mkdir(parents=True, exist_ok=True)
                        full_file = exports_folder / filename
                        logger.info("Exporting report...")

                        # Set location
                        self.session.findById("wnd[1]/usr/ctxtDY_PATH").text = str(exports_folder) # path to save the file
                        self.session.findById("wnd[1]/usr/ctxtDY_PATH").caretPosition = 0
                        self.session.findById("wnd[1]/usr/ctxtDY_FILENAME").text = filename
                        logger.info(f"Report saved successfully as {filename}.")

                        # Set the filename
                        self.session.findById("wnd[1]/usr/ctxtDY_FILENAME").caretPosition = 0
                        self.session.findById("wnd[1]/usr/ctxtDY_FILENAME").setFocus()

                        # Press the button to save the file
                        self.session.findById("wnd[1]/tbar[0]/btn[0]").press()

                        time.sleep(2)  # Wait for the file to be saved
                        self.close_sap_security_popup()

                    # Press Enter
                    self.session.findById("wnd[0]").sendVKey(0)

                    #check if the file is saved or exists
                    if full_file.exists():
                        logger.info(f"File {full_file} exists.")
                        return True, full_file
                    else:
                        logger.error(f"File {full_file} does not exist.")
                        return False, None


            else:
                logger.error(f"Element {element_to_wait_for} not found within the timeout.")
                return False, None

        # Handle any exceptions that may occur during the command execution
        except Exception as e:
            logger.error(f"Error during command execution: {str(e)}")

