import pandas as pd


def ingest(data_dir):
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv.zip")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv.zip")

    if orders["order_id"].isna().any() or not orders["order_id"].is_unique:
        raise ValueError("order_id harus terisi dan unik.")

    return orders.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
