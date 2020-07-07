import psycopg2
import sys
import io

#what to pull out of database
pull_GF = """select "Year", "Day", "Hour", "Minute", "Bx", "By", "Bz", "V", "n", "AE", "AL", "AU", "SYM_H" from test LIMIT 60"""
#what to insert into the database
insert_GF = """ INSERT INTO test ("AE", "AL", "AU", "SYM_H") VALUES (%s,%s,%s,%s)"""

#move this to a config file later
param_dic = {
    "host"      : "postgres",
    "port"      : 5432,
    "database"  : "mydb",
    "user"      : "root",
    "password"  : "example"
    
}

def postgres(params_dic):
    '''Establish a connection to the database by creating a cursor object'''
    try:
        conn = psycopg2.connect(**params_dic)
        cur = conn.cursor()# Create a cursor object
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
    return conn, cur


def posgres_pull(task:str):
        # A sample query of all data from the "vendors" table in the "suppliers" database
    conn, cur = postgres(param_dic)
    try:
        cur.execute(task) #what is your reqest from sql
        colname = [desc[0] for desc in cur.description] #list of column names. we may need this later
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

def postgres_insert(task:str, results):
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





#results = posgres_pull(pull_GF)


#postgres_insert(insert_GF, results)

