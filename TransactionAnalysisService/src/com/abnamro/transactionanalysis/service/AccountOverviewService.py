import psycopg2
import csv

from src.com.abnamro.transactionanalysis.constants import Constants


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


# Example usage:
if __name__ == "__main__":
    # Database configuration for PostgreSQL
    db_config = {
        'host': Constants.HOST,
        'port': Constants.PORT,
        'dbname': Constants.DATABASE_NAME,
        'user': Constants.USER_NAME,
        'password': Constants.PASSWORD
    }

    # Initialize the account overview service
    account_service = AccountOverviewService(db_config)

    # Connect to the database
    account_service.connect()

    # Insert a new record into the account_overview table
    """
    account_service.insert_record(account_number="NL72ABNA054758761", transaction_date="20240329", indicator="C",
                                  amount=3000.00, payment_method="SEPA OVERBOEKING", merchant="Unknown",
                                  description_lines="SEPA Overboeking                 IBAN: NL21ABNA0249151187        BIC: ABNANL2A                    Naam: SS SINGH",
                                  expense_type="Others", expense_category="M")
                                  
                                  """
    # Path to the CSV file
    csv_file_path = Constants.CSV_FILE_PATH

    # Insert records from the CSV file into the account_overview table
    account_service.insert_from_csv(csv_file_path)

    # Close the connection to the database
    account_service.close_connection()
