import psycopg2
import csv

from com.abnamro.llm.constants import Constants


class AccountOverviewService:
    def __init__(self, db_config):
        """
        Initializes the service with database configuration.

        :param db_config: Dictionary containing database configuration with keys:
                          host, port, dbname, user, and password.
        """
        self.db_config = db_config
        self.connection = None

    def connect(self):
        """
        Establishes a connection to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(**self.db_config)
            print("Connection to PostgreSQL DB successful")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection = None

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        if self.connection:
            self.connection.close()
            print("Connection closed")

    def insert_record(self, account_number, transaction_date, indicator, amount, payment_method, merchant,
                      description_lines, expense_type, expense_category):
        """
            Inserts a record into the account_overview table.

            :param expense_category:
            :param expense_type:
            :param merchant:
            :param payment_method:
            :param amount: Float representing the amount sep in the account.
            :param indicator:
            :param transaction_date:
            :param account_number: string representing the account number
            :param description_lines:
        """
        if not self.connection:
            print("No database connection. Please connect first.")
            return

        try:
            cursor = self.connection.cursor()
            insert_query = Constants.SQL_QUERY_INSERT_INTO_ACCOUNT_OVERVIEW
            print(Constants.SQL_QUERY_INSERT_INTO_ACCOUNT_OVERVIEW)
            cursor.execute(insert_query, (
                account_number, transaction_date, indicator, amount, payment_method, merchant, description_lines,
                expense_type, expense_category))

            # Commit the transaction
            self.connection.commit()

            print(f"Record inserted successfully for account_number: {account_number}")
            cursor.close()
        except Exception as e:
            if self.connection:
                self.connection.rollback()  # Rollback in case of error
            print(f"Error inserting record: {e}")

    def insert_from_csv(self, csv_file_path):
        """
        Reads a CSV file and inserts records into the account_overview table.

        :param csv_file_path: Path to the CSV file.
        """
        print(csv_file_path)

        if not self.connection:
            print("No database connection. Please connect first.")
            return

        try:
            with open(csv_file_path, mode='r') as csvfile:

                reader = csv.DictReader(csvfile)

                for row in reader:
                    account_number = row['ï»¿AccountNumber']
                    transaction_date = row['Date']
                    indicator = row['D-C-ind']
                    amount = float(row['Amount'])
                    payment_method = row['Payment Method']
                    merchant = row['Merchant']
                    description_lines = row['Description Lines']
                    expense_type = row['ExpenseType']
                    expense_category = row['ExpenseCategory']

                    # Insert each record into the table
                    self.insert_record(account_number, transaction_date, indicator, amount, payment_method, merchant,
                                       description_lines, expense_type, expense_category)

            print("All records from CSV inserted successfully")
        except Exception as e:
            print(f"Error reading CSV file or inserting records: {e}")
        finally:
            self.close_connection()
