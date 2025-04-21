# Cocoa (ICCO), International Cocoa Organization daily price
# USD / kg

select substr(month,1,4) Year, round(avg(price),2) average_price
from prices
group by substr(month,1,4)
order by 1 desc

# Month with the biggest percentage change
select month, max(Change)
from Prices



select *
from Prices order by 3 desc

delete from prices