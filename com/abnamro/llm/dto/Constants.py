DATABASE_NAME = "postgres"
USER_NAME = "postgres"
PASSWORD = "postgres"
HOST = "localhost"
PORT = 5432

SELECT_ALL_FROM_CATEGORY_TABLE = "select * from expense_category"

SQL_QUERY_INSERT_INTO_ACCOUNT_OVERVIEW = """INSERT INTO account_overview (account_number, transaction_date, 
\"indicator\", amount, payment_method, merchant, description_lines, expense_type, expense_category) values(%s, %s, %s, %s, %s, %s, %s, 
%s, %s)"""

SQL_QUERY_SELECT_ALL_ACCOUNT_OVERVIEW = "SELECT * FROM account_overview"

SQL_QUERY_SELECT_ACCOUNT_OVERVIEW_WHERE_ACCOUNT_NUMBER = "select * from account_overview where account_number=%s%"
