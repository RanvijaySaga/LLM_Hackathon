import psycopg2

from com.abnamro.llm.dto import Constants

global self


# establishing the connection
def __init__(slef):
    self.conn = psycopg2.connect(database=Constants.DATABASE_NAME, user=Constants.USER_NAME,
                                 password=Constants.PASSWORD, port=Constants.PORT)


# get database connection
def getConnection():
    return self.conn


# get cursor from database connection
def getCursor():
    # Creating a cursor object using the cursor() method
    self.cursor = self.conn.cursor()
    return self.cursor


# execute select query
def executeSelectQuery(query):
    # Executing an MYSQL function using the execute() method
    cursor = getCursor()
    cursor.execute(query)

    # Fetch a single row using fetchone() method.
    result = cursor.fetchall()
    final_result = [i[0] for i in result]
    return final_result


def closeConnection():
    # Closing the connection
    self.conn.close()
