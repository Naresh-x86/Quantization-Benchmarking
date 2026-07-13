import json

# Mock Database
ORDERS = {
    "ORD-992": {
        "user_id": 1045,
        "item_id": "ITEM-ZB-PRO",
        "item_name": "ZenBook Pro",
        "status": "delivered",
        "price": 1499.99
    }
}

INVENTORY = {
    "ITEM-ZB-PRO": {
        "stock": 5,
        "estimated_shipping_days": 5
    }
}

POLICIES = {
    "damaged_item": "If an item arrives damaged, the customer is entitled to a full refund or a free replacement. Replacements are subject to inventory shipping times. If the customer requires the item sooner than the replacement shipping time, process a refund.",
    "refunds": "Refunds are processed immediately and take 3-5 business days to reflect on the customer's account."
}


def search_policies(query: str) -> str:
    """
    Search the company policies for a given query (e.g., 'damaged item', 'refunds').
    """
    query = query.lower()
    for key, policy in POLICIES.items():
        if key in query or query in key:
            return f"Policy on {key}: {policy}"
    return "No matching policy found."


def check_order(order_id: str) -> str:
    """
    Check the details of an order using the order ID.
    """
    if order_id in ORDERS:
        return json.dumps(ORDERS[order_id])
    return f"Order {order_id} not found."


def check_inventory(item_id: str) -> str:
    """
    Check inventory and shipping estimates for an item ID.
    """
    if item_id in INVENTORY:
        return json.dumps(INVENTORY[item_id])
    return f"Item {item_id} not found in inventory."


def issue_refund(order_id: str, reason: str) -> str:
    """
    Issue a full refund for a given order ID.
    """
    if order_id in ORDERS:
        return f"SUCCESS: Refund of ${ORDERS[order_id]['price']} issued for {order_id}. Reason: {reason}"
    return f"FAILED: Order {order_id} not found."


def issue_replacement(order_id: str, item_id: str) -> str:
    """
    Issue a replacement for a given order ID and item ID.
    """
    if order_id in ORDERS and item_id in INVENTORY:
        if INVENTORY[item_id]['stock'] > 0:
            INVENTORY[item_id]['stock'] -= 1
            return f"SUCCESS: Replacement ordered for {order_id}. Estimated arrival in {INVENTORY[item_id]['estimated_shipping_days']} days."
        else:
            return "FAILED: Item out of stock."
    return "FAILED: Invalid order or item ID."


AVAILABLE_TOOLS = {
    "search_policies": search_policies,
    "check_order": check_order,
    "check_inventory": check_inventory,
    "issue_refund": issue_refund,
    "issue_replacement": issue_replacement
}

def get_tools_description() -> str:
    return """
Available Tools:
1. search_policies(query: str) -> str
   - Description: Search the company policies for a given query (e.g., 'damaged item').
2. check_order(order_id: str) -> str
   - Description: Check the details of an order using the order ID. Returns JSON.
3. check_inventory(item_id: str) -> str
   - Description: Check inventory and shipping estimates for an item ID. Returns JSON.
4. issue_refund(order_id: str, reason: str) -> str
   - Description: Issue a full refund for a given order ID.
5. issue_replacement(order_id: str, item_id: str) -> str
   - Description: Issue a replacement for a given order ID and item ID.
"""
