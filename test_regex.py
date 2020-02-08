


import re
pattern = r'<subject>(.*)<\/subject>(<content>(.*)<\/content>)?<maincat>(.*)<\/maincat>'
text = '<subject>How to unclog a tub?</subject><maincat>Home</maincat>'
m = re.match(pattern, text)

print(m.group(1))