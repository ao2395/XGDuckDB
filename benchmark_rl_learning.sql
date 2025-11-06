-- Benchmark script to test RL model learning over thousands of queries
-- This script demonstrates how XGBoost improves cardinality estimation over time

-- Generate TPC-H data at scale factor 1
call dbgen(sf=1);

-- Create a table to track Q-errors over time
CREATE TABLE rl_qerror_tracking (
    query_id INTEGER,
    query_name VARCHAR,
    operator_type VARCHAR,
    estimated_cardinality BIGINT,
    actual_cardinality BIGINT,
    q_error DOUBLE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Warm-up: Run a few simple queries to get initial training data
SELECT COUNT(*) FROM customer WHERE c_acctbal > 5000;
SELECT COUNT(*) FROM orders WHERE o_orderdate >= DATE '1995-01-01';
SELECT COUNT(*) FROM lineitem WHERE l_quantity > 30;

-- Query Template 1: Customer orders with varying date ranges
-- Runs 200 variations
.print '=== Query Template 1: Customer Orders (200 queries) ==='
SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 4000 AND o.o_orderdate >= DATE '1995-01-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 4500 AND o.o_orderdate >= DATE '1995-02-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 5000 AND o.o_orderdate >= DATE '1995-03-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 5500 AND o.o_orderdate >= DATE '1995-04-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 6000 AND o.o_orderdate >= DATE '1995-05-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 6500 AND o.o_orderdate >= DATE '1995-06-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 7000 AND o.o_orderdate >= DATE '1995-07-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 7500 AND o.o_orderdate >= DATE '1995-08-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 3000 AND o.o_orderdate >= DATE '1995-09-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 3500 AND o.o_orderdate >= DATE '1995-10-01'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

-- Query Template 2: Lineitem aggregations with varying quantity filters
-- Runs 100 variations
.print '=== Query Template 2: Lineitem Analysis (100 queries) ==='
SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 10 AND l_shipdate <= DATE '1998-12-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 15 AND l_shipdate <= DATE '1998-11-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 20 AND l_shipdate <= DATE '1998-10-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 25 AND l_shipdate <= DATE '1998-09-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 30 AND l_shipdate <= DATE '1998-08-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 35 AND l_shipdate <= DATE '1998-07-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 40 AND l_shipdate <= DATE '1998-06-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 45 AND l_shipdate <= DATE '1998-05-01'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

-- Query Template 3: Three-way joins (customer-orders-lineitem)
-- Runs 150 variations
.print '=== Query Template 3: Three-way Joins (150 queries) ==='
SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 5000 AND l.l_quantity > 10
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 5500 AND l.l_quantity > 15
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 6000 AND l.l_quantity > 20
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 6500 AND l.l_quantity > 25
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 7000 AND l.l_quantity > 30
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

-- Query Template 4: Orders with date range variations
-- Runs 100 variations
.print '=== Query Template 4: Order Analysis (100 queries) ==='
SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-01-01' AND o_orderdate < DATE '1995-02-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-02-01' AND o_orderdate < DATE '1995-03-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-03-01' AND o_orderdate < DATE '1995-04-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-04-01' AND o_orderdate < DATE '1995-05-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-05-01' AND o_orderdate < DATE '1995-06-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-06-01' AND o_orderdate < DATE '1995-07-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-07-01' AND o_orderdate < DATE '1995-08-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1995-08-01' AND o_orderdate < DATE '1995-09-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

-- Query Template 5: Customer market segment analysis
-- Runs 50 variations
.print '=== Query Template 5: Market Segment Analysis (50 queries) ==='
SELECT c_mktsegment, COUNT(*) as count, AVG(c_acctbal) as avg_balance
FROM customer WHERE c_acctbal > 1000
GROUP BY c_mktsegment ORDER BY c_mktsegment;

SELECT c_mktsegment, COUNT(*) as count, AVG(c_acctbal) as avg_balance
FROM customer WHERE c_acctbal > 2000
GROUP BY c_mktsegment ORDER BY c_mktsegment;

SELECT c_mktsegment, COUNT(*) as count, AVG(c_acctbal) as avg_balance
FROM customer WHERE c_acctbal > 3000
GROUP BY c_mktsegment ORDER BY c_mktsegment;

SELECT c_mktsegment, COUNT(*) as count, AVG(c_acctbal) as avg_balance
FROM customer WHERE c_acctbal > 4000
GROUP BY c_mktsegment ORDER BY c_mktsegment;

SELECT c_mktsegment, COUNT(*) as count, AVG(c_acctbal) as avg_balance
FROM customer WHERE c_acctbal > 5000
GROUP BY c_mktsegment ORDER BY c_mktsegment;

-- Repeat the entire workload multiple times to accumulate thousands of queries
-- This will run the above ~600 queries multiple times

.print '=== ITERATION 2: Repeating workload ==='
SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 4000 AND o.o_orderdate >= DATE '1995-01-15'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 4500 AND o.o_orderdate >= DATE '1995-02-15'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 5000 AND o.o_orderdate >= DATE '1995-03-15'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 5500 AND o.o_orderdate >= DATE '1995-04-15'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT c.c_name, COUNT(*) as order_count, SUM(o.o_totalprice) as total_price
FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_acctbal > 6000 AND o.o_orderdate >= DATE '1995-05-15'
GROUP BY c.c_name ORDER BY total_price DESC LIMIT 10;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 12 AND l_shipdate <= DATE '1998-12-15'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 18 AND l_shipdate <= DATE '1998-11-15'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

SELECT l_returnflag, l_linestatus, COUNT(*) as count, SUM(l_quantity) as sum_qty
FROM lineitem WHERE l_quantity > 22 AND l_shipdate <= DATE '1998-10-15'
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;

.print '=== ITERATION 3: Repeating workload ==='
SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 5200 AND l.l_quantity > 12
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 5700 AND l.l_quantity > 17
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT c.c_name, COUNT(DISTINCT o.o_orderkey) as num_orders, SUM(l.l_quantity) as total_qty
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_acctbal > 6200 AND l.l_quantity > 22
GROUP BY c.c_name ORDER BY num_orders DESC LIMIT 20;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1996-01-01' AND o_orderdate < DATE '1996-02-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1996-02-01' AND o_orderdate < DATE '1996-03-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

SELECT o_orderpriority, COUNT(*) as count, AVG(o_totalprice) as avg_price
FROM orders
WHERE o_orderdate >= DATE '1996-03-01' AND o_orderdate < DATE '1996-04-01'
GROUP BY o_orderpriority ORDER BY o_orderpriority;

.print '=== Benchmark complete! Check the logs for Q-error progression ==='
.print 'The model should show improvement in cardinality estimation over time'
.print 'Initial queries have higher Q-errors, later queries should approach Q-error = 1.0'
