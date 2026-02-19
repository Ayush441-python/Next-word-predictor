import re
def clean_text(x):
    x = re.sub('[^a-zA-Z]',' ',x)
    x = re.sub('<.*?>',' ',x)
    return x.strip()


# ayu = '''kmmjnhbtvr7n8jb65v478
# j786h5g
# j78h6g5
# k8jhg
# n98j7bh6
# mk89nj7bh6g
# k89njb7hvg
# 9n8b76g
# mo987n6546ikbjyvthjmku
# bnmikt
# bnmik
# htb4678mo8kujhrv5p.;/.lhp[.['p.,jbhyghujikop,;o.k,jmhtg6h6o;.k,jmjhgyuhjkop;,.,jjhtygu6io
# l,j jbtuhjioml,mhtbhhuik]]'''

# print(clean_text(ayu))


