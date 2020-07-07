import psycopg2
import sys
import io
import time


#what to pull out of database #25 min for rolling
pull_GF = """select "Year", "Day", "Hour", "Minute", "Bx", "By", "Bz", "V", "n", "AE", "AL", "AU", "SYM_H" from wind LIMIT 10000"""
#pull_GF = """select "Year", "Day", "Hour", "Minute", "Bx", "By", "Bz", "v", "n", "ae", "al", "au", "sym_h" from wind LIMIT 48"""
#what to insert into the database ORDER BY timestamp DESC 
insert_GF = """ INSERT INTO globalfc ("Year", "Day", "Hour", "Minute", "AE", "AL", "AU", "SYM_H", "timestamp") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
#wee need to work on syncing data that is rolling in. I will just make this do one pull then stop. because its a data fump
create_table1 = '''create table globalfc
(
    "Year"      integer,
    "Day"       integer,
    "Hour"      integer,
    "Minute"    integer,
    "AE"        float,
    "AL"        float,
    "AU"        float,
    "SYM_H"     float,
    "timestamp" integer
)'''

create_table2 = '''create table wind
(
    "Year"      integer,
    "Day"       integer,
    "Hour"      integer,
    "Minute"    integer,
    "Bx"        NUMERIC (7,2),
    "By"        NUMERIC (7,2),
    "Bz"        NUMERIC (7,2),
    "V"         NUMERIC (7,2),
    "n"         NUMERIC (7,2),
    "AE"        integer,
    "AL"        integer,
    "AU"        integer,
    "SYM_H"     integer
)'''

#move this to a config file later
param_dic = {
    "host"      : "localhost", 
    "port"      : 5432,
    "database"  : "astradb",
    "user"      : "admin1",
    "password"  : "admin1"
    }
#host: "postgres",


def postgres(params_dic):
    '''Establish a connection to the database by creating a cursor object'''
    try:
        conn = psycopg2.connect(**params_dic)
        cur = conn.cursor()# Create a cursor object
    except (Exception, psycopg2.DatabaseError) as error:
        time.sleep(60) #waits 60 seconds then try again
        postgres(param_dic) #continously trying to connect
        #print(error)
        #sys.exit(1) 
    return conn, cur

def create_table(sql, params_dic):
    """creating tables"""
    conn, cur = postgres(param_dic)
    try:
        cur.execute(sql) #what is your reqest from sql
        conn.commit()
    except (Exception, psycopg2.Error) as error : #exception for failing to connect
        if(conn):
            print("Failed to insert record into mobile table", error)
    finally:
        #closing database connection.
        if(conn):
            cur.close()
            conn.close()
            #print("PostgreSQL connection is closed")


def posgres_pull(task:str, params_dic):
        # A sample query of all data from the "vendors" table in the "suppliers" database
    conn, cur = postgres(param_dic)
    try:
        cur.execute(task) #what is your reqest from sql
        #colname = [desc[0] for desc in cur.description] #list of column names. we may need this later
        #print(colname)
        query_results = cur.fetchall()
        results = query_results
        cur.close() # Close the cursor and connection to so the server can allocate bandwidth to other requests
        conn.close()
        return results
    except (Exception, psycopg2.Error) as error : #exception for failing to connect
        if(conn):
            print("Failed to insert record into mobile table", error)
    finally:
        #closing database connection.
        if(conn):
            cur.close()
            conn.close()
            #print("PostgreSQL connection is closed")

def postgres_insert(task:str, results, params_dic):
    """ Execute a single INSERT request """
    conn, cur = postgres(param_dic)
    try:
        for d in results:
            cur.execute(task, d)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cur.close()
        return 1
    cur.close()
    conn.close()




create_table(create_table1, param_dic)
create_table(create_table2, param_dic)
#results = posgres_pull(pull_GF, param_dic)


#postgres_insert(insert_GF, results)


