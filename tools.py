import json

# --- AGENT 1: CUSTOMER SUPPORT TOOLS ---
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
        "in_stock": True,
        "shipping_days": 5
    }
}

def check_order(order_id: str) -> str:
    if order_id in ORDERS:
        return json.dumps(ORDERS[order_id])
    return "Order not found."

def check_inventory(item_id: str) -> str:
    if item_id in INVENTORY:
        return json.dumps(INVENTORY[item_id])
    return "Item not found in inventory."

def issue_replacement(order_id: str) -> str:
    if order_id in ORDERS:
        return f"Replacement issued for order {order_id}."
    return "Order not found."

def issue_refund(order_id: str) -> str:
    if order_id in ORDERS:
        return f"Refund of ${ORDERS[order_id]['price']} issued for order {order_id}."
    return "Order not found."

# --- AGENT 2: IT HELPDESK TOOLS ---
SERVICES = {
    "api_gateway": "online",
    "auth_service": "online",
    "payment_backend": "down",
    "database": "online"
}

LOGS = {
    "payment_backend": "ERROR: Connection refused to database port 5432. Process crashed."
}

def check_server_status() -> str:
    return json.dumps(SERVICES)

def read_service_logs(service_name: str) -> str:
    if service_name in LOGS:
        return LOGS[service_name]
    return f"No logs found for {service_name}."

def restart_service(service_name: str) -> str:
    if service_name in SERVICES:
        SERVICES[service_name] = "online"
        return f"Service {service_name} has been successfully restarted."
    return "Service not found."

# --- AGENT 3: FINANCIAL ANALYST TOOLS ---
STOCK_DATA = {
    "TECH_CORP": {
        "current_price": 145.50,
        "5_day_history": [152.0, 150.5, 149.0, 147.5, 145.5]
    }
}

def get_current_price(ticker: str) -> str:
    if ticker in STOCK_DATA:
        return f"Current price of {ticker} is ${STOCK_DATA[ticker]['current_price']}"
    return "Ticker not found."

def get_5_day_average(ticker: str) -> str:
    if ticker in STOCK_DATA:
        history = STOCK_DATA[ticker]["5_day_history"]
        avg = sum(history) / len(history)
        return f"5-day average for {ticker} is ${avg:.2f}"
    return "Ticker not found."

def execute_trade(action: str, ticker: str, shares: int) -> str:
    if action not in ["buy", "sell"]:
        return "Invalid action. Must be 'buy' or 'sell'."
    if ticker in STOCK_DATA:
        return f"Successfully executed {action} of {shares} shares for {ticker}."
    return "Ticker not found."

# --- TOOL REGISTRIES ---
SUPPORT_TOOLS = {
    "check_order": check_order,
    "check_inventory": check_inventory,
    "issue_replacement": issue_replacement,
    "issue_refund": issue_refund
}

IT_TOOLS = {
    "check_server_status": check_server_status,
    "read_service_logs": read_service_logs,
    "restart_service": restart_service
}

FINANCE_TOOLS = {
    "get_current_price": get_current_price,
    "get_5_day_average": get_5_day_average,
    "execute_trade": execute_trade
}

def get_tools_description(agent_id: str) -> str:
    tools_dict = get_tools_dict(agent_id)
    desc = ""
    for name in tools_dict:
        desc += f"- {name}\n"
    return desc

def get_tools_dict(agent_id: str) -> dict:
    if agent_id == "AGENT_1_SUPPORT":
        return SUPPORT_TOOLS
    elif agent_id == "AGENT_2_IT_HELPDESK":
        return IT_TOOLS
    elif agent_id == "AGENT_3_FINANCE":
        return FINANCE_TOOLS
    return {}
