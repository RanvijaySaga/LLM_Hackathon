import psycopg2
from psycopg2 import sql
from datetime import datetime

from com.abnamro.llm.dto import Constants


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
    account_service.insert_record(account_number="NL72ABNA054758761", transaction_date="20240329", indicator="C",
                                  amount=3000.00, payment_method="SEPA OVERBOEKING", merchant="Unknown",
                                  description_lines="SEPA Overboeking                 IBAN: NL21ABNA0249151187        BIC: ABNANL2A                    Naam: SS SINGH",
                                  expense_type="Others",expense_category="M")

    # Close the connection to the database
    account_service.close_connection()
