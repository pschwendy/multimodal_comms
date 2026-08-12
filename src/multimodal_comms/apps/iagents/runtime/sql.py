import mysql.connector
import mysql.connector.pooling
import os
import yaml
import logging

file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(project_path, "config/global.yaml"), "r"))

DATABASE_USER = os.getenv("MYSQL_USER", global_config.get("mysql").get("username"))
DATABASE_PASSWD = os.getenv("MYSQL_PASSWORD", global_config.get("mysql").get("password"))
if os.getenv("DOCKERIZED"):
    HOST = "db"
else:
    HOST = os.getenv("MYSQL_HOST", global_config.get("mysql").get("host"))
DATABASE = os.getenv("MYSQL_DATABASE", global_config.get("mysql").get("database"))
PORT = int(os.getenv("MYSQL_PORT", global_config.get("mysql").get("port", 3306)))

DB_CONNECT_TIMEOUT = 300

# MySQL database configuration
db_config = {
    'host': HOST,
    'port': PORT,
    'user': DATABASE_USER,
    'password': str(DATABASE_PASSWD),
    'database': DATABASE,
    'connect_timeout': DB_CONNECT_TIMEOUT,
    'raise_on_warnings': True,
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}

DB_POOL_NAME = "my_pool"
DB_POOL_SIZE = 20

db_pool = None


def get_db_pool():
    """Create the MySQL pool on first database use, not during app import."""
    global db_pool
    if db_pool is None:
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name=DB_POOL_NAME, pool_size=DB_POOL_SIZE, **db_config
        )
    return db_pool


def execute_sql(sql_command, conn, cursor, params=None):
    try:
        conn.ping(reconnect=True, attempts=3, delay=2)
        if params:
            cursor.execute(sql_command, params)
        else:
            cursor.execute(sql_command)
    except mysql.connector.Error as err:
        logging.error("Error executing SQL command: {}".format(err))
        cursor.close()
        cursor = conn.cursor(buffered=True)
        cursor.execute(sql_command)
    return cursor


def exec_sql(sql_command, params=None, mode="read"):
    conn = get_db_pool().get_connection()
    cursor = conn.cursor(buffered=True)
    try:
        cursor = execute_sql(sql_command, conn, cursor, params)
        if mode == "write":
            conn.commit()
            return "write success"
        else:
            result = cursor.fetchall() or []
            return result
    finally:
        cursor.close()
        conn.close()
