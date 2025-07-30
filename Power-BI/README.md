# Financial Report

This Power BI report provides a 3-page sales dashboard for a fictional retail company designed to support both business and technical analysis, built using a clean star schema and advanced DAX.

## Features
- KPIs Overview (Sales, Units Sold, Margin, Cost)
- DAX Measures
- Customer Segmentation by Discount Band and Segment
- Time Intelligence & Interactive Filters

## Key Insights
- Germany accounted for nearly 45% of total sales, making it the top market.
- Sales peaked in June and December, suggesting strong seasonal buying trends.
- The Public Sector segment consistently led sales across all countries.
- City Bikes emerged as the top-selling product category overall.

## Technical Details
- Star Schema data model
- DAX measures for KPIs, rankings, and time intelligence
- Power Query for data transformation and cleaning
- Complete Date Table created using [Devin Knight’s guide](https://www.sqlchick.com/entries/creating-a-date-dimension-table-in-power-bi)

## Preview
![Overview Page](overview_page.png?raw=true "Overview Page")

## The Star Schema
![Star schema](star_schema.png?raw=true "Star schema")

## DAX Measures

- Total Sales = 
  SUM('Fact Sales'[ Sales ])

- Total Units Sold =
  SUM('Fact Sales'[Units Sold])

- Profit = 
  SUM('Fact Sales'[ Profit ])

- MoM Sales Growth % =
  VAR CurrentSales = [Total Sales]
  VAR LastMonthSales =
    CALCULATE ( [Total Sales], DATEADD ( 'Dim Date'[Date], -1, MONTH ) )
  RETURN
    IF (
      ISBLANK ( LastMonthSales ),
      BLANK (),
      DIVIDE ( CurrentSales - LastMonthSales, LastMonthSales )
    )

- Top 5 Sales =
  IF ( [Product Rank] <= 5, [Total Sales], BLANK () )

- Product Rank =
  RANKX ( ALL ( 'Dim Product'[ Product ] ), [Total Sales],, DESC )

- Segment Sales % Share =
  DIVIDE (
    [Total Sales],
    CALCULATE ( [Total Sales], ALL ( 'Dim Customer'[Segment] ) ),
    0
  )

## Notes

- Built entirely in Power BI Desktop

- Based on fictional data for demonstration purposes

- Clean star schema model with clearly defined fact and dimension tables

## How to Use
- Open the .pbix file in Power BI Desktop to explore the visuals, model, and DAX logic.
- Hold Ctrl and click to navigate between report pages
