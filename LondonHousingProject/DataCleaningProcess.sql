-- Convert the 'date' column to a standard date format in both tables
UPDATE monthly_data
SET date = STR_TO_DATE(date, '%Y-%m-%d');

UPDATE yearly_data
SET date = STR_TO_DATE(date, '%Y-%m-%d');

-- Example: Fill missing 'no_of_crimes' with 0 in 'monthly_data'
UPDATE monthly_data
SET no_of_crimes = COALESCE(no_of_crimes, 0);

-- For numeric columns like 'median_salary', consider setting to the average or median
-- First, find the average or median (this is an example, adjust according to your needs)
SELECT AVG(median_salary) FROM yearly_data WHERE median_salary IS NOT NULL;

-- Then, update missing values with this average (replace 'your_average_here' with the actual value)
UPDATE yearly_data
SET median_salary = COALESCE(median_salary, your_average_here);

-- Remove duplicates in 'monthly_data'
DELETE m1 FROM monthly_data m1
INNER JOIN monthly_data m2 
WHERE 
    m1.id > m2.id AND 
    m1.date = m2.date AND 
    m1.area = m2.area;

-- Repeat for 'yearly_data' if necessary

-- Convert 'recycling_pct' to numeric in 'yearly_data'
-- This assumes the column is stored as text and contains only numeric values or percentages
UPDATE yearly_data
SET recycling_pct = CAST(recycling_pct AS UNSIGNED INTEGER);

-- Delete rows where the date is before 2010
DELETE FROM monthly_data WHERE date < '2010-01-01';
DELETE FROM yearly_data WHERE date < '2010-01-01';

-- Create a view that joins the two datasets based on area and year
CREATE VIEW combined_data AS
SELECT 
    m.*, y.median_salary, y.mean_salary, y.recycling_pct, y.population_size, y.number_of_jobs, y.area_size, y.no_of_houses
FROM 
    monthly_data m
INNER JOIN 
    yearly_data y ON m.area = y.area AND YEAR(m.date) = YEAR(y.date);

-- Example: If there was a column 'new_build' with values 'Y' or 'N'
UPDATE london_housing
SET new_build = CASE WHEN new_build = 'Y' THEN 'Yes'
                     WHEN new_build = 'N' THEN 'No'
                     ELSE new_build
                     END;

-- Example: Remove duplicate rows based on 'area', 'date', and 'average_price'
WITH CTE AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY area, date, average_price
    ORDER BY date) AS rn
  FROM london_housing
)
DELETE FROM CTE WHERE rn > 1;

-- Example: Drop the 'no_of_crimes' column if it was deemed unnecessary
ALTER TABLE london_housing
DROP COLUMN no_of_crimes;
