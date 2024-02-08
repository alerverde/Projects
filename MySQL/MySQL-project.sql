/* Lest begin verifiying that the data was properly loaded */ 

SELECT table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'bank'
ORDER BY 1;


/* We can also see all the different types of constraints */

SELECT constraint_name, table_name, constraint_type
FROM information_schema.table_constraints
WHERE table_schema = 'bank'
ORDER BY 3,1;


/* List the customers (id and city) who have at least one business associated with them */ 

SELECT c.cust_id, c.state 
FROM customer c WHERE EXISTS ( SELECT 1 
    FROM business b
    WHERE b.cust_id = c.cust_id );


/* Find the employees with the highest total transaction amount */ 

SELECT e.emp_id, e.fname, e.lname, SUM(t.amount) AS total_amount 
FROM employee e
    INNER JOIN account a ON a.open_emp_id = e.emp_id
    INNER JOIN transaction t ON a.account_id=t.account_id 
GROUP BY e.emp_id 
ORDER BY total_amount;


/* List the departments where all employees (teller and head teller) have made at least one transaction */ 


SELECT d.dept_id, d.name
FROM department d
WHERE NOT EXISTS (
    SELECT 1
    FROM employee e
    LEFT JOIN account a ON e.emp_id = a.open_emp_id
    WHERE d.dept_id = e.dept_id
    AND (e.title = 'Teller' OR e.title='Head Teller')
    AND a.account_id IS NULL
);


/* List employees with a flag indicating whether they have worked with a customer from a different city */

SELECT e.emp_id, e.fname, e.lname,
    CASE
        WHEN (
            SELECT COUNT(DISTINCT c.city)
            FROM customer c  
            INNER JOIN account a ON c.cust_id = a.cust_id            
            INNER JOIN transaction t ON a.account_id = t.account_id
            WHERE a.open_emp_id = e.emp_id
        ) > 1 THEN 'Yes'
        ELSE 'No'
    END AS worked_with_different_city
FROM employee e;


/* List departments with an indication of whether they have employees managing accounts */

SELECT d.dept_id, d.name,
    CASE
        WHEN EXISTS ( SELECT 1
            FROM account a
            INNER JOIN employee e ON a.open_emp_id = e.emp_id
            WHERE e.dept_id = d.dept_id
        ) THEN 'Yes'
        ELSE 'No'
    END AS has_account_manager
FROM department d;


/* Show customers and their total account balances, marking those with balances exceeding the average */

SELECT c.cust_id, SUM(a.avail_balance) AS total_balance,
    CASE
        WHEN SUM(a.avail_balance) > (
            SELECT AVG(avail_balance)
            FROM account
        ) THEN 'Exceeds Average'
        ELSE 'Within Average'
    END AS balance_status
FROM customer c
INNER JOIN account a ON c.cust_id = a.cust_id
GROUP BY c.cust_id;


