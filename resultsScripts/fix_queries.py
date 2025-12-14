import re

def fix_sql():
    print("Reading queries_sf10.sql...")
    with open('queries_sf10.sql', 'r') as f:
        content = f.read()

    print("Applying fixes...")
    # Remove 'top X' (case insensitive)
    # Note: We need to be careful not to match 'select top ...' if it's not there, but the regex handles it.
    # We also want to keep the 'select' part.
    # The previous regex replaced 'select top X ' with 'select '.
    content = re.sub(r'(?i)select\s+top\s+\d+\s+', 'select ', content)

    # Replace '+/- X days' with '+/- INTERVAL 'X' DAY'
    content = re.sub(r'([+-])\s+(\d+)\s+days', r"\1 INTERVAL '\2' DAY", content)

    # Quote 'at' alias to avoid syntax error
    content = re.sub(r'\) at,', ') "at",', content)
    
    # Quote 'returns' alias to avoid syntax error
    content = re.sub(r'\) returns', ') "returns"', content)
    content = re.sub(r' as returns', ' as "returns"', content)
    content = re.sub(r'coalesce\(returns, 0\) returns', 'coalesce(returns, 0) "returns"', content)

    print("Writing back to queries_sf10.sql...")
    with open('queries_sf10.sql', 'w') as f:
        f.write(content)
    print("Done.")

if __name__ == "__main__":
    fix_sql()
