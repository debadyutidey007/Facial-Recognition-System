-- Create the main database if it doesn't exist
CREATE DATABASE IF NOT EXISTS FACE_ANALYSIS;

-- Use the database
USE FACE_ANALYSIS;

-- Create user_accounts table with proper column naming
CREATE TABLE IF NOT EXISTS USER_ACCOUNTS (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    USERNAME VARCHAR(50) NOT NULL UNIQUE,
    PASSWORD_HASH VARCHAR(255) NOT NULL,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create analysis_results table with proper column naming
-- Using the "gender_prediction" column name instead of "gender" as requested
CREATE TABLE IF NOT EXISTS ANALYSIS_RESULTS (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    USER_ID INT NOT NULL,
    IMAGE_FILENAME VARCHAR(255) NOT NULL,
    EMOTION VARCHAR(50) NOT NULL,
    GENDER_PREDICTION VARCHAR(50) NOT NULL,
    AGE INT NOT NULL,
    ETHNICITY VARCHAR(50) NOT NULL,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (USER_ID) REFERENCES USER_ACCOUNTS(ID) ON DELETE CASCADE
);

-- Create indexes for faster queries
CREATE INDEX IDX_USER_ID ON ANALYSIS_RESULTS(USER_ID);

CREATE INDEX IDX_USERNAME ON USER_ACCOUNTS(USERNAME);

CREATE INDEX IDX_CREATED_AT ON ANALYSIS_RESULTS(CREATED_AT);

-- Grant permissions (adjust as needed for your MySQL setup)
-- GRANT ALL PRIVILEGES ON face_analysis_db.* TO 'your_user'@'localhost';
-- FLUSH PRIVILEGES;

-- Sample queries for common operations

-- Get all users
-- SELECT * FROM user_accounts;

-- Get user by username
-- SELECT * FROM user_accounts WHERE username = 'sample_username';

-- Get analysis history for a specific user (most recent first)
-- SELECT * FROM analysis_results WHERE user_id = 1 ORDER BY created_at DESC;

-- Clear history for a specific user
-- DELETE FROM analysis_results WHERE user_id = 1;

-- Delete a user and their related analysis results (cascade delete will handle the analysis_results)
-- DELETE FROM user_accounts WHERE id = 1;