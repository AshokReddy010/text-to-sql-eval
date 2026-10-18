# Business glossary

The analytics team's agreed definitions. Questions use these terms exactly as defined here.

- **Today** is 2026-06-30. "This year" is 2026; "last year" is 2025. Years and months are calendar periods.
- **Valid order**: an order whose `status` is not `'cancelled'`. Cancelled orders never count toward revenue, order counts, customers' order history or averages unless a question asks about cancellations.
- **Revenue** (net sales): `SUM(quantity * unit_price * (1 - discount))` over `order_items` of valid orders. Refunds are *not* subtracted.
- **Net revenue after refunds**: revenue minus `SUM(returns.refund_amount)` for returns on those orders.
- **Cost of goods**: `SUM(quantity * products.unit_cost)` over the same order items.
- **Gross margin %**: `(revenue - cost of goods) / revenue * 100`.
- **Order value**: the revenue of a single valid order. **Average order value (AOV)**: revenue divided by the number of valid orders.
- **Active customer**: a customer with at least one valid order dated on or after 2026-04-01 (the last 91 days up to today).
- **Repeat customer**: a customer with two or more valid orders.
- **Return rate %**: orders with at least one row in `returns`, divided by orders whose status is `'delivered'` or `'returned'`, times 100.
- **Days to ship**: `ship_date - order_date` in days, for orders that have a `ship_date`.
- **Referred customer**: a customer whose `referred_by` is not NULL; the referrer is the customer with that `customer_id`.
- **Percentages** are on a 0–100 scale. When a question says "rounded", round only the final result.
