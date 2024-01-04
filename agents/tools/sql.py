import sqlite3
from pydantic import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")

def list_tables():
    c = conn.cursor();
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"

class RunQueryArgsSchema(BaseModel):
    query: str

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",  # LLM uses this description to help it know when to run this tool
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema  # tells LLM to provide a property called "query" of type "string"
)

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

class DescribeTablesArgSchema(BaseModel):
    table_names: List[str]

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, return a schema of those tables.",
    func=describe_tables,
    args_schema=DescribeTablesArgSchema
)