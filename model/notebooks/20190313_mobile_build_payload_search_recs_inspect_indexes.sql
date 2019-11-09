-- To inspect tables and indexes
.mode line
select * from sqlite_master;

-- To query tables/indexes with total size
select name, sum(pgsize) from dbstat group by name;
