DROP TABLE IF EXISTS Results;
DROP TABLE IF EXISTS Jobs;
DROP TABLE IF EXISTS Batches;



-- Create the Batches table
CREATE TABLE Batches (
    batch_id INT IDENTITY(1,1) PRIMARY KEY,
    name NVARCHAR(255),
    description NVARCHAR(MAX),
    created_at DATETIME DEFAULT GETDATE()
);

-- Create the Jobs table
CREATE TABLE Jobs (
    job_id INT IDENTITY(1,1) PRIMARY KEY,
    status NVARCHAR(50) DEFAULT 'PENDING',
    
    instance_name NVARCHAR(255), 
    batch_id INT, 
    capacity INT,     
    alpha FLOAT,
    seed BIGINT,
    iterations INT,
    is_time_based BIT DEFAULT 0,
    base_time FLOAT NULL,
    time_limit FLOAT NULL,
    customer_count INT NULL,
    min_customer_count INT NULL,
    rcl_strategy NVARCHAR(50),
    local_search_operators NVARCHAR(255),
    construction_strategy NVARCHAR(50) DEFAULT 'insertion',
    local_search_strategy NVARCHAR(50) DEFAULT 'sequential',
    
    demands_data NVARCHAR(MAX),
    coords_data NVARCHAR(MAX),
    
    created_at DATETIME DEFAULT GETDATE(),
    started_at DATETIME NULL,
    completed_at DATETIME NULL,
    error_message NVARCHAR(MAX)

FOREIGN KEY (batch_id) REFERENCES Batches(batch_id) ON DELETE SET NULL);

-- Create the Results table
CREATE TABLE Results (
    result_id INT IDENTITY(1,1) PRIMARY KEY,
    job_id INT,
    solve_time FLOAT,
    objective_value FLOAT,
    vehicle_count INT,
    avg_load_factor FLOAT,
    best_known_cost FLOAT,
    gap_to_bks FLOAT,
    best_known_vehicle_count INT,
    gap_to_bks_vehicle_count FLOAT,

    routes NVARCHAR(MAX),
    bks_routes NVARCHAR(MAX),
    history NVARCHAR(MAX),
    
    executed_at DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (job_id) REFERENCES Jobs(job_id) ON DELETE CASCADE
);



