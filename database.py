import psycopg2
import psycopg2.extras
import os
import json
import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Database configuration from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

def dict_factory(cursor, row):
    """Convert database row to dictionary with column names as keys."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def initialize_database():
    """Create the database tables if they don't exist."""
    conn = None
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Create analysis_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            algorithm TEXT NOT NULL,
            threshold REAL NOT NULL,
            image_size_kb REAL NOT NULL,
            change_percentage REAL NOT NULL,
            ssim REAL NOT NULL,
            psnr REAL NOT NULL,
            emd REAL NOT NULL,
            description TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Initialize database when module is imported
db_initialized = initialize_database()

def save_analysis_result(result_data: Dict[str, Any]) -> int:
    """
    Save analysis result to the database.
    
    Args:
        result_data (Dict[str, Any]): Dictionary containing analysis result data
            - algorithm (str): Name of the algorithm used
            - threshold (float): Threshold value used
            - image_size_kb (float): Size of the processed images in KB
            - change_percentage (float): Percentage of pixels that changed
            - ssim (float): Structural Similarity Index value
            - psnr (float): Peak Signal-to-Noise Ratio value
            - emd (float): Earth Mover's Distance value
            - description (str, optional): User-provided description
            - timestamp (datetime, optional): Timestamp of the analysis
    
    Returns:
        int: ID of the saved record, or -1 if there was an error
    """
    if not db_initialized:
        print("Database not initialized")
        return -1
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Convert timestamp to string if it's a datetime object
        if isinstance(result_data.get('timestamp'), datetime.datetime):
            result_data['timestamp'] = result_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare the query
        query = '''
        INSERT INTO analysis_results
        (algorithm, threshold, image_size_kb, change_percentage, ssim, psnr, emd, description, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        '''
        
        # Convert NumPy types to Python native types
        threshold = float(result_data.get('threshold', 0.0))
        image_size_kb = float(result_data.get('image_size_kb', 0.0))
        change_percentage = float(result_data.get('change_percentage', 0.0))
        ssim_value = float(result_data.get('ssim', 0.0))
        psnr_value = float(result_data.get('psnr', 0.0))
        emd_value = float(result_data.get('emd', 0.0))
        
        # Extract values from the result_data dictionary
        values = (
            result_data.get('algorithm', ''),
            threshold,
            image_size_kb,
            change_percentage,
            ssim_value,
            psnr_value,
            emd_value,
            result_data.get('description', ''),
            result_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        
        # Execute the query
        cursor.execute(query, values)
        result_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return the ID of the inserted record
        return result_id
    except Exception as e:
        print(f"Error saving analysis result: {e}")
        if conn:
            conn.rollback()
        return -1
    finally:
        if conn:
            conn.close()

def get_analysis_results(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent analysis results from the database.
    
    Args:
        limit (int): Maximum number of results to retrieve
    
    Returns:
        List[Dict[str, Any]]: List of analysis results as dictionaries
    """
    if not db_initialized:
        print("Database not initialized")
        return []
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = '''
        SELECT * FROM analysis_results
        ORDER BY timestamp DESC
        LIMIT %s
        '''
        
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        
        # Convert DictRows to regular dictionaries
        dict_results = []
        for row in results:
            dict_results.append(dict(row))
            
        # Convert timestamp strings back to datetime objects
        for result in dict_results:
            if 'timestamp' in result and not isinstance(result['timestamp'], datetime.datetime):
                result['timestamp'] = datetime.datetime.strptime(
                    str(result['timestamp']), '%Y-%m-%d %H:%M:%S'
                )
        
        return dict_results
    except Exception as e:
        print(f"Error retrieving analysis results: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_algorithm_performance() -> List[Dict[str, Any]]:
    """
    Calculate performance metrics for each algorithm from the database.
    
    Returns:
        List[Dict[str, Any]]: List of algorithm performance metrics
            Each dictionary contains:
            - algorithm (str): Name of the algorithm
            - analysis_count (int): Number of analyses performed with this algorithm
            - avg_ssim (float): Average SSIM value
            - avg_psnr (float): Average PSNR value
            - avg_emd (float): Average EMD value
            - avg_change_percentage (float): Average percentage of pixels that changed
    """
    if not db_initialized:
        print("Database not initialized")
        return []
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = '''
        SELECT 
            algorithm,
            COUNT(*) as analysis_count,
            AVG(ssim) as avg_ssim,
            AVG(psnr) as avg_psnr,
            AVG(emd) as avg_emd,
            AVG(change_percentage) as avg_change_percentage
        FROM analysis_results
        GROUP BY algorithm
        ORDER BY analysis_count DESC
        '''
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert DictRows to regular dictionaries
        dict_results = []
        for row in results:
            dict_results.append(dict(row))
            
        return dict_results
    except Exception as e:
        print(f"Error retrieving algorithm performance: {e}")
        return []
    finally:
        if conn:
            conn.close()
