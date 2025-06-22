import os
import logging
import mysql.connector
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from crewai import Agent, Task, Crew, BaseLLM, LLM
from pydantic import BaseModel, Field
import json
import re
from datetime import datetime
import traceback
from dotenv import load_dotenv
import time
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import for Replicate LLM (assuming you have litellm or similar)
try:
    from litellm import completion
except ImportError:
    logger.warning("litellm not installed. Please install it: pip install litellm")
    completion = None

class ReplicateLLM(BaseLLM):
    def __init__(self, model: str, temperature: float | None = None):
        super().__init__(model=model, temperature=temperature)
        self.model = model
    
    @property
    def _llm_type(self):
        return "Replicate"
    
    def call(self,
             messages: str | list[dict[str, str]],
             tools=None,
             callbacks=None,
             available_functions=None,
             ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if tools and self.supports_function_calling():
            response = completion(self.model, messages=messages, tools=tools)
        else:
            response = completion(self.model, messages=messages)
        return response["choices"][0]["message"]["content"]
    
    def supports_function_calling(self) -> bool:
        return True

class DatabaseConfig:
    """Database configuration class"""
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.username = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "adminpass")
        self.database = os.getenv("DB_NAME", "sakila")
        self.port = int(os.getenv("DB_PORT", 3306))

class SQLConnectionTool:
    """Enhanced SQL connection and execution tool with improved error handling"""
    
    def __init__(self):
        self.name = "sql_connection_tool"
        self.description = "Connects to MySQL database and executes SQL queries with comprehensive error handling"
        self.db_config = DatabaseConfig()
        self.connection = None
        self.schema_info = None
        self._initialize_connection()
        self._load_schema_info()
    
    def _initialize_connection(self):
        """Initialize database connection with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.connection = mysql.connector.connect(
                    host=self.db_config.host,
                    user=self.db_config.username,
                    password=self.db_config.password,
                    database=self.db_config.database,
                    port=self.db_config.port,
                    autocommit=True,
                    connection_timeout=30,
                    charset='utf8mb4'
                )
                logger.info("Database connection established successfully")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)
    
    def _load_schema_info(self):
        """Load comprehensive database schema information"""
        try:
            cursor = self.connection.cursor()
            
            # Get detailed table information with sample data
            cursor.execute("""
            SELECT 
                t.TABLE_NAME,
                t.TABLE_COMMENT,
                c.COLUMN_NAME, 
                c.DATA_TYPE, 
                c.IS_NULLABLE, 
                c.COLUMN_KEY, 
                c.COLUMN_COMMENT,
                c.COLUMN_DEFAULT,
                c.EXTRA
            FROM INFORMATION_SCHEMA.TABLES t
            JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
            WHERE t.TABLE_SCHEMA = %s AND c.TABLE_SCHEMA = %s
            ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
        """, (self.db_config.database, self.db_config.database))
            
            schema_data = cursor.fetchall()
            
            # Get foreign key relationships
            cursor.execute("""
                SELECT 
                    kcu.TABLE_NAME,
                    kcu.COLUMN_NAME,
                    kcu.REFERENCED_TABLE_NAME,
                    kcu.REFERENCED_COLUMN_NAME,
                    rc.CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc 
                    ON kcu.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
                WHERE kcu.TABLE_SCHEMA = %s 
                AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
            """, (self.db_config.database,))
            
            fk_data = cursor.fetchall()
            
            # Get table row counts for context
            table_counts = {}
            cursor.execute(f"SHOW TABLES FROM {self.db_config.database}")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    table_counts[table_name] = count
                except:
                    table_counts[table_name] = 0
            
            # Organize schema information
            self.schema_info = {
                'tables': {},
                'foreign_keys': [],
                'table_counts': table_counts,
                'database_name': self.db_config.database
            }
            
            # Process schema data
            for row in schema_data:
                table_name, table_comment, column_name, data_type, is_nullable, column_key, column_comment, default_val, extra = row
                
                if table_name not in self.schema_info['tables']:
                    self.schema_info['tables'][table_name] = {
                        'comment': table_comment or '',
                        'columns': [],
                        'row_count': table_counts.get(table_name, 0)
                    }
                
                self.schema_info['tables'][table_name]['columns'].append({
                    'column': column_name,
                    'type': data_type,
                    'nullable': is_nullable,
                    'key': column_key,
                    'comment': column_comment or '',
                    'default': default_val,
                    'extra': extra
                })
            
            # Process foreign keys
            for fk in fk_data:
                self.schema_info['foreign_keys'].append({
                    'table': fk[0],
                    'column': fk[1],
                    'ref_table': fk[2],
                    'ref_column': fk[3],
                    'constraint_name': fk[4]
                })
            
            cursor.close()
            logger.info("Comprehensive schema information loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load schema information: {str(e)}")
            self.schema_info = {'tables': {}, 'foreign_keys': [], 'table_counts': {}}
    
    def get_schema_description(self) -> str:
        """Get comprehensive formatted schema description for the AI agent"""
        if not self.schema_info:
            return "Schema information not available"
        
        description = f"DATABASE: {self.schema_info.get('database_name', 'Unknown')}\n\n"
        description += "=== DETAILED SCHEMA INFORMATION ===\n\n"
        
        # Add table information with context
        for table_name, table_info in self.schema_info['tables'].items():
            row_count = table_info.get('row_count', 0)
            table_comment = table_info.get('comment', '')
            
            description += f"Table: {table_name} ({row_count:,} rows)\n"
            if table_comment:
                description += f"Description: {table_comment}\n"
            
            description += "Columns:\n"
            for col in table_info['columns']:
                key_info = f" [{col['key']}]" if col['key'] else ""
                nullable = "NULL" if col['nullable'] == 'YES' else "NOT NULL"
                default = f" DEFAULT {col['default']}" if col['default'] else ""
                comment = f" -- {col['comment']}" if col['comment'] else ""
                
                description += f"  - {col['column']}: {col['type']}{key_info} {nullable}{default}{comment}\n"
            description += "\n"
        
        # Add foreign key relationships
        if self.schema_info['foreign_keys']:
            description += "=== FOREIGN KEY RELATIONSHIPS ===\n"
            for fk in self.schema_info['foreign_keys']:
                description += f"  {fk['table']}.{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}\n"
            description += "\n"
        
        # Add some business context examples
        description += "=== QUERY EXAMPLES FOR REFERENCE ===\n"
        description += "- For counting records: SELECT COUNT(*) FROM table_name\n"
        description += "- For aggregations: SELECT column, COUNT(*), SUM(amount) FROM table GROUP BY column\n"
        description += "- For joins: SELECT * FROM table1 t1 JOIN table2 t2 ON t1.id = t2.foreign_id\n"
        description += "- Always use table aliases for complex queries\n"
        description += "- Use LIMIT for large result sets\n\n"

        # Add Sakila-specific business context
        description += "=== SAKILA DATABASE SPECIFIC NOTES ===\n"
        description += "- rental_rate and replacement_cost are in the 'film' table, NOT 'rental' table\n"
        description += "- To calculate revenue: use film.rental_rate or payment.amount\n"
        description += "- rental table contains rental dates and return dates\n"
        description += "- payment table contains actual payment amounts\n"
        description += "- For revenue analysis, JOIN rental -> payment for actual amounts\n"
        description += "- For potential revenue, use film.rental_rate\n\n"
        
        return description
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query with comprehensive error handling and validation"""
        max_retries = 3
        result = {
            'success': False,
            'data': None,
            'error': None,
            'row_count': 0,
            'execution_time': 0,
            'query': query.strip()
        }
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Validate query safety - IMPROVED VERSION
                safety_check = self._is_safe_query(query)
                if not safety_check['is_safe']:
                    result['error'] = f"Query safety check failed: {safety_check['reason']}"
                    return result
                
                # Check connection
                if not self.connection or not self.connection.is_connected():
                    self._initialize_connection()
                
                cursor = self.connection.cursor()
                cursor.execute(query)
                
                # Handle SELECT queries
                if query.strip().upper().startswith('SELECT') or query.strip().upper().startswith('WITH'):
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    result['success'] = True
                    result['row_count'] = len(results)
                    
                    if results:
                        # Convert to DataFrame for better handling
                        df = pd.DataFrame(results, columns=columns)
                        result['data'] = df
                    else:
                        result['data'] = pd.DataFrame(columns=columns)
                        result['error'] = "Query executed successfully but returned no results"
                else:
                    # Handle non-SELECT queries
                    result['success'] = True
                    result['row_count'] = cursor.rowcount
                    result['error'] = f"Query executed successfully. Rows affected: {cursor.rowcount}"
                
                cursor.close()
                break
                
            except mysql.connector.Error as e:
                error_msg = str(e)
                logger.error(f"SQL execution attempt {attempt + 1} failed: {error_msg}")
                
                # Provide helpful error suggestions
                if "doesn't exist" in error_msg.lower():
                    if "rental_rate" in error_msg:
                        result['error'] = f"Column error: rental_rate is in 'film' table, not 'rental' table. Use film.rental_rate or payment.amount for revenue calculations."
                    else:
                        result['error'] = f"Table or column doesn't exist: {error_msg}. Please check the schema."
                elif "syntax error" in error_msg.lower():
                    result['error'] = f"SQL syntax error: {error_msg}. Please check your query syntax."
                elif "access denied" in error_msg.lower():
                    result['error'] = f"Access denied: {error_msg}. Check your database permissions."
                else:
                    result['error'] = f"Database error: {error_msg}"
                
                if attempt == max_retries - 1:
                    break
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Unexpected error in attempt {attempt + 1}: {str(e)}")
                result['error'] = f"Unexpected error: {str(e)}"
                if attempt == max_retries - 1:
                    break
                time.sleep(1)
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _is_safe_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query safety validation with improved logic"""
        
        # Clean up the query for analysis
        query_clean = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove comments
        query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)  # Remove block comments
        query_clean = query_clean.strip().upper()
        
        # Define dangerous operations that should be blocked
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 
            'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE', 'FLUSH', 'RESET',
            'CALL', 'EXECUTE', 'PREPARE', 'DEALLOCATE'
        ]
        
        # Check for dangerous keywords at word boundaries
        for keyword in dangerous_keywords:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_clean):
                return {
                    'is_safe': False,
                    'reason': f"Query contains dangerous operation: {keyword}"
                }
        
        # Check for SQL injection patterns
        injection_patterns = [
            r';\s*DROP\s+',
            r';\s*DELETE\s+',
            r';\s*INSERT\s+',
            r';\s*UPDATE\s+',
            r'UNION\s+ALL\s+SELECT.*FROM\s+INFORMATION_SCHEMA',
            r'1\s*=\s*1',
            r'OR\s+1\s*=\s*1',
            r'\';\s*--',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query_clean):
                return {
                    'is_safe': False,
                    'reason': f"Query contains potential SQL injection pattern"
                }
        
        # Check for multiple statements (semicolon followed by non-whitespace)
        semicolon_count = len(re.findall(r';\s*\S', query_clean))
        if semicolon_count > 0:
            return {
                'is_safe': False,
                'reason': "Query contains multiple statements"
            }
        
        # Allow WITH clauses and SELECT statements
        if query_clean.startswith('WITH') or query_clean.startswith('SELECT'):
            return {'is_safe': True, 'reason': 'Query is safe'}
        
        # If we get here, it's likely safe but be cautious
        return {'is_safe': True, 'reason': 'Query passed safety checks'}

class Text2SQLAgent:
    """Enhanced Text2SQL agent using CrewAI with dynamic query generation"""
    
    def __init__(self):
        self.sql_tool = SQLConnectionTool()
        # Use Gemini as the default LLM - it showed good performance in your logs
        # self.llm = LLM(
        #     model="gemini/gemini-2.5-flash-preview-04-17",
        #     temperature=0.3
        # )
         # self.llm = ReplicateLLM(model="replicate/meta/meta-llama-3-8b-instruct")
        # self.llm = LLM(
        #     model="groq/llama-3.3-70b-versatile",
        #     temperature=0.3
        # )
        self.llm = LLM(
            model="gemini/gemini-2.5-flash-preview-04-17",
            temperature=0.3
        )
        self.setup_agents()
    
    def setup_agents(self):
        """Setup CrewAI agents with enhanced roles and LLM integration"""
        
        # SQL Query Generator Agent
        self.sql_generator = Agent(
            role="Senior SQL Developer",
            goal="Generate accurate, optimized SQL queries from natural language questions",
            backstory="""You are a senior database developer with 10+ years of experience in MySQL. 
            You excel at understanding complex business questions and translating them into efficient SQL queries.
            You always consider performance, readability, and correctness. You use appropriate JOINs, 
            GROUP BY clauses, and aggregate functions. You never make assumptions about data that isn't 
            explicitly provided in the schema. You are especially skilled with CTEs (Common Table Expressions) 
            and window functions for complex analytical queries.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Query Validator and Optimizer Agent
        self.sql_validator = Agent(
            role="Database Performance Specialist",
            goal="Validate, optimize, and ensure SQL query correctness and performance",
            backstory="""You are a database performance expert who specializes in query optimization 
            and validation. You review SQL queries for syntax errors, performance issues, and logical 
            correctness. You suggest improvements like proper indexing usage, efficient JOINs, and 
            optimal WHERE clauses. You ensure queries are safe and won't harm the database. You are 
            particularly skilled at identifying missing WITH clauses and CTE syntax issues.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Data Analyst and Insights Agent
        self.data_analyst = Agent(
            role="Senior Data Analyst",
            goal="Analyze query results and provide meaningful business insights",
            backstory="""You are a senior data analyst with expertise in business intelligence and 
            data interpretation. You can quickly identify patterns, trends, and anomalies in data. 
            You provide clear, actionable insights and suggest appropriate visualizations for 
            different types of data. You understand business contexts and can relate data findings 
            to real-world scenarios.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def process_question(self, user_question: str) -> Dict[str, Any]:
        """Process user question with comprehensive error handling and retry logic"""
        
        logger.info(f"Processing question: {user_question}")
        
        try:
            # Step 1: Generate SQL Query
            sql_generation_result = self._generate_sql_with_llm(user_question)
            
            if not sql_generation_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to generate SQL query: {sql_generation_result['error']}",
                    'question': user_question
                }
            
            sql_query = sql_generation_result['sql_query']
            
            # Step 2: Validate and optimize the query
            validation_result = self._validate_and_optimize_query(sql_query, user_question)
            
            if validation_result['optimized_query']:
                sql_query = validation_result['optimized_query']
            
            # Step 3: Execute the query with retry mechanism
            execution_result = self._execute_with_retry(sql_query)
            
            if not execution_result['success']:
                # Try to fix the query if execution failed
                fixed_query_result = self._attempt_query_fix(sql_query, execution_result['error'], user_question)
                if fixed_query_result['success']:
                    sql_query = fixed_query_result['sql_query']
                    execution_result = self._execute_with_retry(sql_query)
            
            # Step 4: Analyze results if query was successful
            analysis_result = None
            if execution_result['success'] and execution_result['data'] is not None:
                analysis_result = self._analyze_results(execution_result['data'], user_question)
            
            return {
                'success': execution_result['success'],
                'question': user_question,
                'sql_query': sql_query,
                'execution_result': execution_result,
                'validation_notes': validation_result.get('notes', ''),
                'analysis': analysis_result,
                'dataframe': execution_result.get('data'),
                'visualization_data': self._prepare_visualization_data(
                    execution_result.get('data'), user_question
                ) if execution_result['success'] else None
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'question': user_question
            }
    
    def _generate_sql_with_llm(self, user_question: str) -> Dict[str, Any]:
        """Generate SQL query using LLM with comprehensive schema context"""
        
        try:
            schema_description = self.sql_tool.get_schema_description()
            
            prompt = f"""
You are an expert SQL developer. Generate a MySQL query to answer the following question.

QUESTION: {user_question}

{schema_description}

INSTRUCTIONS:
1. Generate ONLY the SQL query, no explanations
2. Use proper MySQL syntax
3. Use table aliases for readability
4. Include appropriate JOINs based on foreign key relationships
5. Ensure the query directly answers the question asked
6. Use aggregate functions (COUNT, SUM, AVG, etc.) when appropriate
7. For complex queries requiring multiple steps, use WITH clauses (CTEs)
8. Do NOT include any INSERT, UPDATE, DELETE, or DROP statements
9. Always end with a semicolon
10. For queries involving rankings or "top N" results, use appropriate window functions or ORDER BY with LIMIT

RESPONSE FORMAT:
Return only the SQL query without any markdown formatting or explanations.
The query should be ready to execute.
"""
            
            # Create SQL generation task
            sql_task = Task(
                description=prompt,
                expected_output="A valid MySQL SELECT query that answers the user's question",
                agent=self.sql_generator
            )
            
            # Execute the task
            crew = Crew(
                agents=[self.sql_generator],
                tasks=[sql_task],
                verbose=False
            )
            
            result = crew.kickoff()
            
            # Extract SQL query from result
            sql_query = self._extract_sql_from_response(str(result))
            
            if sql_query:
                return {
                    'success': True,
                    'sql_query': sql_query,
                    'raw_response': str(result)
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not extract valid SQL query from LLM response',
                    'raw_response': str(result)
                }
                
        except Exception as e:
            logger.error(f"Error generating SQL with LLM: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """Extract SQL query from LLM response with improved logic"""
        try:
            # Remove common markdown formatting
            response = response.strip()
            
            # Remove ```sql or ``` blocks
            if '```' in response:
                parts = response.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('sql'):
                        part = part[3:].strip()
                    if ('SELECT' in part.upper() or 'WITH' in part.upper()) and part:
                        response = part.strip()
                        break
            
            # Look for SQL statement
            lines = response.split('\n')
            sql_lines = []
            capturing = False
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith('SELECT') or line.upper().startswith('WITH'):
                    capturing = True
                    sql_lines.append(line)
                elif capturing:
                    if line and not line.startswith('--') and not line.startswith('#'):
                        sql_lines.append(line)
                        if line.endswith(';'):
                            break
            
            if sql_lines:
                sql_query = ' '.join(sql_lines)
                # Clean up the query
                sql_query = re.sub(r'\s+', ' ', sql_query).strip()
                if not sql_query.endswith(';'):
                    sql_query += ';'
                return sql_query
            
            # Fallback: if the entire response looks like a query
            if ('SELECT' in response.upper() or 'WITH' in response.upper()) and len(response.split()) > 3:
                sql_query = re.sub(r'\s+', ' ', response).strip()
                if not sql_query.endswith(';'):
                    sql_query += ';'
                return sql_query
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting SQL from response: {str(e)}")
            return None
    
    def _validate_and_optimize_query(self, sql_query: str, user_question: str) -> Dict[str, Any]:
        """Validate and optimize SQL query using validator agent"""
        
        try:
            validation_prompt = f"""
Review and optimize this SQL query for the question: "{user_question}"

SQL QUERY TO REVIEW:
{sql_query}

VALIDATION CHECKLIST:
1. Syntax correctness (especially WITH clause syntax)
2. Performance optimization opportunities
3. Proper use of JOINs and relationships
4. Appropriate use of indexes
5. Query safety (no dangerous operations)
6. Logic correctness for answering the question

If the query is syntactically correct and logically sound, respond with "QUERY_APPROVED".
If optimization is needed, provide the improved query starting with "OPTIMIZED:" followed by the query.
If there are syntax errors, provide the corrected query starting with "CORRECTED:" followed by the query.
"""
            
            validation_task = Task(
                description=validation_prompt,
                expected_output="Query validation feedback and optimized/corrected query if needed",
                agent=self.sql_validator
            )
            
            crew = Crew(
                agents=[self.sql_validator],
                tasks=[validation_task],
                verbose=False
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            # Check if optimization was provided
            optimized_query = None
            if "QUERY_APPROVED" not in result_str:
                if "OPTIMIZED:" in result_str:
                    optimized_query = self._extract_query_after_keyword(result_str, "OPTIMIZED:")
                elif "CORRECTED:" in result_str:
                    optimized_query = self._extract_query_after_keyword(result_str, "CORRECTED:")
                else:
                    optimized_query = self._extract_sql_from_response(result_str)
            
            return {
                'success': True,
                'notes': result_str,
                'optimized_query': optimized_query
            }
            
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'notes': 'Validation failed, using original query'
            }
    
    def _extract_query_after_keyword(self, text: str, keyword: str) -> Optional[str]:
        """Extract SQL query that appears after a specific keyword"""
        try:
            if keyword in text:
                parts = text.split(keyword, 1)
                if len(parts) > 1:
                    query_part = parts[1].strip()
                    return self._extract_sql_from_response(query_part)
            return None
        except Exception as e:
            logger.error(f"Error extracting query after keyword: {str(e)}")
            return None
    
    def _execute_with_retry(self, sql_query: str) -> Dict[str, Any]:
        """Execute query with retry mechanism"""
        return self.sql_tool.execute_query(sql_query)
    
    def _attempt_query_fix(self, failed_query: str, error_message: str, user_question: str) -> Dict[str, Any]:
        """Attempt to fix a failed query using LLM"""
        
        try:
            fix_prompt = f"""
The following SQL query failed with an error. Please fix it.

ORIGINAL QUESTION: {user_question}
FAILED QUERY: {failed_query}
ERROR MESSAGE: {error_message}

{self.sql_tool.get_schema_description()}

Common fixes needed:
1. Add missing WITH keyword for CTEs
2. Fix column references
3. Correct JOIN syntax
4. Fix GROUP BY clauses

Please provide a corrected SQL query that addresses the error.
Return only the corrected SQL query, properly formatted and ready to execute.
"""
            
            fix_task = Task(
                description=fix_prompt,
                expected_output="A corrected SQL query that fixes the identified error",
                agent=self.sql_generator
            )
            
            crew = Crew(
                agents=[self.sql_generator],
                tasks=[fix_task],
                verbose=False
            )
            
            result = crew.kickoff()
            fixed_query = self._extract_sql_from_response(str(result))
            
            if fixed_query:
                return {
                    'success': True,
                    'sql_query': fixed_query
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not generate a fixed query'
                }
                
        except Exception as e:
            logger.error(f"Error attempting query fix: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_results(self, df: pd.DataFrame, user_question: str) -> Dict[str, Any]:
        """Analyze query results using data analyst agent"""
        
        try:
            if df is None or df.empty:
                return {
                    'success': False,
                    'message': 'No data to analyze'
                }
            
            # Prepare data summary
            data_summary = f"""
Data Summary:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {', '.join(df.columns.tolist())}

Sample Data (first 3 rows):
{df.head(3).to_string()}

Data Types:
{df.dtypes.to_string()}
"""
            
            if len(df) > 3:
                data_summary += f"\n\nSample Data (last 3 rows):\n{df.tail(3).to_string()}"
            
            analysis_prompt = f"""
Analyze the results for the question: "{user_question}"

{data_summary}

Provide:
1. Key findings and insights
2. Notable patterns or trends
3. Business implications
4. Recommendations for data visualization
5. Any anomalies or interesting observations

Be concise but insightful.
"""
            
            analysis_task = Task(
                description=analysis_prompt,
                expected_output="Comprehensive data analysis with business insights",
                agent=self.data_analyst
            )
            
            crew = Crew(
                agents=[self.data_analyst],
                tasks=[analysis_task],
                verbose=False
            )
            
            result = crew.kickoff()
            
            return {
                'success': True,
                'insights': str(result),
                'data_summary': data_summary
            }
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_visualization_data(self, df: Optional[pd.DataFrame], user_question: str) -> Dict[str, Any]:
        """Prepare data for visualization with intelligent suggestions"""
        
        if df is None or df.empty:
            return {'type': 'no_data', 'message': 'No data available for visualization'}
        
        viz_data = {
            'dataframe_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict()
            },
            'suggestions': []
        }
        
        # Analyze data types
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Intelligent visualization suggestions
        if len(numeric_cols) >= 2:
            viz_data['suggestions'].append({
                'type': 'scatter',
                'x_axis': numeric_cols[0],
                'y_axis': numeric_cols[1],
                'title': f"{numeric_cols[1]} vs {numeric_cols[0]}"
            })
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            viz_data['suggestions'].append({
                'type': 'bar',
                'x_axis': categorical_cols[0],
                'y_axis': numeric_cols[0],
                'title': f"{numeric_cols[0]} by {categorical_cols[0]}"
            })
        
        if len(categorical_cols) >= 1:
            # Check if suitable for pie chart (not too many categories)
            unique_values = df[categorical_cols[0]].nunique()
            if unique_values <= 10:
                viz_data['suggestions'].append({
                    'type': 'pie',
                    'category_col': categorical_cols[0],
                    'title': f"Distribution of {categorical_cols[0]}"
                })
        
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            viz_data['suggestions'].append({
                'type': 'line',
                'x_axis': datetime_cols[0],
                'y_axis': numeric_cols[0],
                'title': f"{numeric_cols[0]} over time"
            })
        
        # Default to table if no specific visualization is suitable
        if not viz_data['suggestions']:
            viz_data['suggestions'].append({
                'type': 'table',
                'title': 'Query Results',
                'message': 'Displaying results in table format'
            })
        
        viz_data['sample_data'] = df.head(100).to_dict('records')  # Limit for performance
        
        return viz_data

class VisualizationEngine:
    """Enhanced visualization engine with multiple chart types"""
    
    def __init__(self):
        self.chart_types = ['bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'table']
    
    def create_visualization(self, df: pd.DataFrame, viz_config: Dict[str, Any]) -> go.Figure:
        """Create visualization based on configuration"""
        
        if df is None or df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        chart_type = viz_config.get('type', 'table')
        
        try:
            if chart_type == 'bar':
                return self._create_bar_chart(df, viz_config)
            elif chart_type == 'line':
                return self._create_line_chart(df, viz_config)
            elif chart_type == 'scatter':
                return self._create_scatter_plot(df, viz_config)
            elif chart_type == 'pie':
                return self._create_pie_chart(df, viz_config)
            elif chart_type == 'histogram':
                return self._create_histogram(df, viz_config)
            elif chart_type == 'box':
                return self._create_box_plot(df, viz_config)
            else:
                return self._create_table_view(df, viz_config)
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return self._create_error_chart(str(e))
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        
        if x_col not in df.columns or y_col not in df.columns:
            return self._create_error_chart("Required columns not found in data")
        
        # Aggregate data if needed
        if df[x_col].dtype == 'object':
            chart_data = df.groupby(x_col)[y_col].sum().reset_index()
            x_data = chart_data[x_col]
            y_data = chart_data[y_col]
        else:
            x_data = df[x_col]
            y_data = df[y_col]
        
        fig = px.bar(
            x=x_data, 
            y=y_data,
            title=config.get('title', f'{y_col} by {x_col}'),
            labels={x_col: x_col.title(), y_col: y_col.title()}
        )
        
        fig.update_layout(
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title(),
            showlegend=False
        )
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        
        if x_col not in df.columns or y_col not in df.columns:
            return self._create_error_chart("Required columns not found in data")
        
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=config.get('title', f'{y_col} over {x_col}')
        )
        
        fig.update_layout(
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title()
        )
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        
        if x_col not in df.columns or y_col not in df.columns:
            return self._create_error_chart("Required columns not found in data")
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            title=config.get('title', f'{y_col} vs {x_col}')
        )
        
        fig.update_layout(
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title()
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create pie chart"""
        category_col = config.get('category_col')
        
        if category_col not in df.columns:
            return self._create_error_chart("Category column not found in data")
        
        # Count occurrences
        value_counts = df[category_col].value_counts()
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=config.get('title', f'Distribution of {category_col}')
        )
        
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create histogram"""
        col = config.get('column')
        
        if col not in df.columns:
            return self._create_error_chart("Column not found in data")
        
        fig = px.histogram(
            df, 
            x=col,
            title=config.get('title', f'Distribution of {col}')
        )
        
        fig.update_layout(
            xaxis_title=col.title(),
            yaxis_title='Count'
        )
        
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create box plot"""
        y_col = config.get('y_axis')
        x_col = config.get('x_axis', None)
        
        if y_col not in df.columns:
            return self._create_error_chart("Y-axis column not found in data")
        
        if x_col and x_col in df.columns:
            fig = px.box(df, x=x_col, y=y_col, title=config.get('title', f'{y_col} by {x_col}'))
        else:
            fig = px.box(df, y=y_col, title=config.get('title', f'Distribution of {y_col}'))
        
        return fig
    
    def _create_table_view(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create table visualization"""
        # Limit rows for performance
        display_df = df.head(100)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[display_df[col].tolist() for col in display_df.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=config.get('title', 'Query Results'),
            height=600
        )
        
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error visualization"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14, color='red')
        )
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
def main():
    """Enhanced Streamlit application with improved UI/UX"""
    
    st.set_page_config(
        page_title="Text2SQL AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"  # Changed from "collapsed" to "expanded"
    )
    
    # Enhanced CSS with better responsive design
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container fixes */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .css-1d391kg .css-1lcbmhc {
        background: transparent;
        border: none;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
        color: white !important;
    }
    
    /* Prevent content shift when sidebar opens/closes */
    .css-1rs6os, .css-17ziqus {
        transition: margin-left 0.3s ease;
    }
    
    /* Header styles */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    
    .input-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
    }
    
    .results-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border: 2px solid #28a745;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
        border: 2px solid #dc3545;
    }
    
    /* Button enhancements */
    .stButton > button {
        font-weight: 500;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.5px;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #495057;
        border: 1px solid #ced4da;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        transform: translateY(-1px);
    }
    
    /* Form improvements */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.5;
        transition: all 0.3s ease;
        resize: vertical;
        min-height: 120px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    .stCheckbox {
        font-weight: 500;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .stProgress .st-bn {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    /* Alert boxes */
    .success-alert {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-alert {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-alert {
        background: linear-gradient(135deg, #cce7ff 0%, #b3d9ff 100%);
        color: #004085;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e9ecef;
        background: #f8f9fa;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #667eea;
        border-right-color: #667eea;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-weight: 500;
    }
    
    /* Sidebar button styling */
    .css-1d391kg .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        width: 100%;
        text-align: left;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
    }
    
    .css-1d391kg .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(2px);
    }
    
    /* Sample questions styling */
    .sample-question-btn {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 1rem !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
    }
    
    .sample-question-btn:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateX(2px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .main-header {
            font-size: 2rem;
        }
        
        .main-subtitle {
            font-size: 1rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header section
    st.markdown('<h1 class="main-header">🤖 Text2SQL AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent' not in st.session_state:
        with st.spinner("🔄 Initializing AI agents and database connection..."):
            try:
                st.session_state.agent = Text2SQLAgent()
                st.session_state.viz_engine = VisualizationEngine()
                st.markdown('<div class="success-alert">✅ System initialized successfully!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-alert">❌ Failed to initialize system: {str(e)}</div>', unsafe_allow_html=True)
                st.stop()
    
    # Initialize session state for user question
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("## 🛠️ Database & Settings")
        
        # Database information card
        if hasattr(st.session_state.agent, 'sql_tool') and st.session_state.agent.sql_tool.schema_info:
            schema_info = st.session_state.agent.sql_tool.schema_info
            
            st.markdown("### 📊 Database Info")
            st.markdown(f"**Database:** {schema_info.get('database_name', 'Unknown')}")
            st.markdown(f"**Tables:** {len(schema_info.get('tables', {}))}")
            
            # Show table summary
            total_rows = sum(table.get('row_count', 0) for table in schema_info.get('tables', {}).values())
            st.markdown(f"**Total Records:** {total_rows:,}")
            
            # Table details in expander
            with st.expander("📋 Table Details"):
                for table_name, table_info in schema_info.get('tables', {}).items():
                    st.markdown(f"**{table_name}**")
                    st.markdown(f"  • Rows: {table_info.get('row_count', 0):,}")
                    st.markdown(f"  • Columns: {len(table_info.get('columns', []))}")
        
        st.markdown("---")
        
        # Sample questions section - FIXED
        st.markdown("### 💡 Sample Questions")
        st.markdown("*Click any question to try it:*")
        
        sample_questions = [
            "How many customers do we have?",
            "Top 10 best-selling films?",
            "Monthly rental trends",
            "Which category has highest revenue?",
            "Customers with most rentals",
            "Average rental duration",
            "Most popular actors",
            "Store performance comparison",
            "Customer activity by city",
            "Seasonal rental patterns"
        ]
        
        # Create buttons for sample questions
        for i, question in enumerate(sample_questions):
            # Use unique keys and proper button handling
            button_key = f"sample_question_{i}"
            if st.button(
                f"📝 {question}", 
                key=button_key,
                help=f"Click to use: {question}",
                use_container_width=True
            ):
                st.session_state.user_question = question
                st.rerun()
    
    # Add sample questions in main area as well (alternative approach)
    st.markdown('<div class="section-header">💡 Quick Start - Sample Questions</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Popular queries to get you started:**")
        
        # Create columns for sample questions in main area
        col1, col2, col3 = st.columns(3)
        
        popular_questions = [
            "How many customers do we have?",
            "Top 10 best-selling films?",
            "Monthly rental trends",
            "Which category has highest revenue?",
            "Customers with most rentals",
            "Average rental duration"
        ]
        
        for i, question in enumerate(popular_questions):
            col_index = i % 3
            if col_index == 0:
                with col1:
                    if st.button(f"📊 {question}", key=f"main_sample_{i}", use_container_width=True):
                        st.session_state.user_question = question
                        st.rerun()
            elif col_index == 1:
                with col2:
                    if st.button(f"📈 {question}", key=f"main_sample_{i}", use_container_width=True):
                        st.session_state.user_question = question
                        st.rerun()
            else:
                with col3:
                    if st.button(f"🔍 {question}", key=f"main_sample_{i}", use_container_width=True):
                        st.session_state.user_question = question
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with improved layout
    st.markdown('<div class="section-header">💬 Ask Your Question</div>', unsafe_allow_html=True)
    
    # Question input section
    with st.container():
        st.markdown('<div class="card input-card">', unsafe_allow_html=True)
        
        # Input area - use session state value
        user_question = st.text_area(
            "Enter your question about the database:",
            value=st.session_state.user_question,
            height=120,
            placeholder="e.g., 'Show me the top 10 customers by total rentals' or 'What are the most popular film categories?'",
            key="question_input",
            help="Ask any question about your database. The AI will generate the appropriate SQL query and provide insights."
        )
        
        # Update session state when text area changes
        if user_question != st.session_state.user_question:
            st.session_state.user_question = user_question
        
        # Control buttons
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 3])
        
        with col1:
            generate_button = st.button("🚀 Generate Query & Analyze", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            if clear_button:
                st.session_state.user_question = ""
                st.rerun()
        
        with col3:
            show_sql = st.checkbox("Show SQL", value=True)
        
        with col4:
            show_analysis = st.checkbox("Show Analysis", value=True)
        
        with col5:
            viz_type = st.selectbox(
                "Visualization:",
                ["Auto-detect", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Table"],
                key="viz_type"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process question when button is clicked
    if generate_button and user_question.strip():
        # Progress section
        st.markdown('<div class="section-header">🤖 AI Processing</div>', unsafe_allow_html=True)
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Analyzing question
                status_text.markdown("🔍 **Analyzing your question...**")
                progress_bar.progress(20)
                
                # Step 2: Generating SQL
                status_text.markdown("⚡ **Generating SQL query...**")
                progress_bar.progress(40)
                
                # Step 3: Executing query
                result = st.session_state.agent.process_question(user_question)
                status_text.markdown("🔄 **Executing query...**")
                progress_bar.progress(70)
                
                # Step 4: Generating insights
                status_text.markdown("📊 **Generating insights and visualizations...**")
                progress_bar.progress(90)
                
                # Step 5: Complete
                status_text.markdown("✅ **Analysis complete!**")
                progress_bar.progress(100)
                
                # Clear progress indicators after a brief moment
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Results section
                st.markdown('<div class="section-header">📊 Results & Insights</div>', unsafe_allow_html=True)
                
                if result['success']:
                    # Success message
                    st.markdown('<div class="success-alert">✅ <strong>Query executed successfully!</strong></div>', unsafe_allow_html=True)
                    
                    # Show SQL query if requested
                    if show_sql and result.get('sql_query'):
                        with st.container():
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### 🔧 Generated SQL Query")
                            st.code(result['sql_query'], language='sql')
                            
                            # Validation notes
                            if result.get('validation_notes'):
                                with st.expander("📝 Query Validation Notes"):
                                    st.write(result['validation_notes'])
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show results
                    df = result.get('dataframe')
                    if df is not None and not df.empty:
                        
                        # Results summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-container"><div class="metric-value">{len(df)}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-container"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
                        with col3:
                            exec_time = result.get('execution_result', {}).get('execution_time', 0)
                            st.markdown(f'<div class="metric-container"><div class="metric-value">{exec_time:.2f}s</div><div class="metric-label">Exec Time</div></div>', unsafe_allow_html=True)
                        
                        # Visualization section
                        st.markdown('<div class="card results-card">', unsafe_allow_html=True)
                        st.markdown("#### 📈 Data Visualization")
                        
                        # Create visualization
                        if result.get('visualization_data'):
                            viz_data = result['visualization_data']
                            
                            # Determine visualization config
                            if viz_type == "Auto-detect" and viz_data.get('suggestions'):
                                viz_config = viz_data['suggestions'][0]
                            else:
                                # Create config based on user selection
                                viz_config = {'type': viz_type.lower().replace(' ', '_').replace('chart', '').replace('plot', '').strip('_')}
                                
                                # Add appropriate columns
                                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                                
                                if viz_config['type'] in ['bar', 'line', 'scatter'] and len(numeric_cols) >= 1:
                                    viz_config['y_axis'] = numeric_cols[0]
                                    if len(categorical_cols) >= 1:
                                        viz_config['x_axis'] = categorical_cols[0]
                                    elif len(numeric_cols) >= 2:
                                        viz_config['x_axis'] = numeric_cols[1]
                                elif viz_config['type'] == 'pie' and len(categorical_cols) >= 1:
                                    viz_config['category_col'] = categorical_cols[0]
                                elif viz_config['type'] == 'histogram' and len(numeric_cols) >= 1:
                                    viz_config['column'] = numeric_cols[0]
                            
                            # Generate and display visualization
                            fig = st.session_state.viz_engine.create_visualization(df, viz_config)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Alternative visualization options
                            if viz_data.get('suggestions') and len(viz_data['suggestions']) > 1:
                                st.markdown("**Try other visualizations:**")
                                viz_cols = st.columns(min(4, len(viz_data['suggestions'])))
                                for i, suggestion in enumerate(viz_data['suggestions'][:4]):
                                    with viz_cols[i]:
                                        if st.button(f"📊 {suggestion.get('title', 'Chart')}", key=f"alt_viz_{i}"):
                                            alt_fig = st.session_state.viz_engine.create_visualization(df, suggestion)
                                            st.plotly_chart(alt_fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # AI Analysis section
                        if show_analysis and result.get('analysis') and result['analysis'].get('success'):
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### 🧠 AI Insights & Analysis")
                            st.markdown(result['analysis']['insights'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Raw data section
                        with st.expander("📊 View Raw Data"):
                            st.dataframe(df, use_container_width=True, height=300)
                            
                            # Download options
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            with col2:
                                # JSON download option
                                json_data = df.to_json(orient='records', indent=2)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_data,
                                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                    
                    else:
                        st.markdown('<div class="warning-alert">⚠️ Query executed successfully but returned no results. Try modifying your question or check if the data exists.</div>', unsafe_allow_html=True)
                
                else:
                    # Error handling
                    st.markdown('<div class="card error-card">', unsafe_allow_html=True)
                    st.markdown("#### ❌ Query Execution Error")
                    st.markdown(f"**Error:** {result.get('error', 'Unknown error occurred')}")
                    
                    if result.get('sql_query'):
                        st.markdown("**Generated SQL Query:**")
                        st.code(result['sql_query'], language='sql')
                        
                        # Error suggestions
                        error_msg = result.get('error', '').lower()
                        if 'syntax' in error_msg:
                            st.markdown("💡 **Suggestion:** There might be a syntax error in the generated query. Try rephrasing your question.")
                        elif 'table' in error_msg or 'column' in error_msg:
                            st.markdown("💡 **Suggestion:** Check if the table or column names exist in your database schema.")
                        elif 'permission' in error_msg:
                            st.markdown("💡 **Suggestion:** You might not have the necessary permissions to access the requested data.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.markdown(f'<div class="error-alert">❌ <strong>Unexpected Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
    
    elif generate_button:
        st.markdown('<div class="warning-alert">⚠️ Please enter a question before generating the query.</div>', unsafe_allow_html=True)
    
    


if __name__ == "__main__":
    main()