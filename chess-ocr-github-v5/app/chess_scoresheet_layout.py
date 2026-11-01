"""Explicit panel ordering without changing recognized move text."""
def order_panels(items, panels, row_sort):
    if panels == 1 or len(items) < panels * 2:
        return row_sort(items)
    # Each panel contains a white and a black move column.
    centers = sorted(set((i["bbox"][0] + i["bbox"][2]) / 2 for i in items))
    count = panels * 2
    if len(centers) < count:
        return row_sort(items)
    gaps = sorted(range(len(centers)-1),
                  key=lambda j: centers[j+1]-centers[j], reverse=True)[:count-1]
    cuts = sorted((centers[j]+centers[j+1])/2 for j in gaps)
    groups = [[] for _ in range(panels)]
    for item in items:
        x = (item["bbox"][0]+item["bbox"][2])/2
        column = sum(x > cut for cut in cuts)
        groups[column // 2].append(item)
    return [item for group in groups for item in row_sort(group)]
