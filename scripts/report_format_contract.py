"""Narrow scientific-preservation exception for the reviewed S7 header fix."""
def require_weighted_header_only(before, after):
    def row(text):
        return [s.strip() for s in text.splitlines() if s.startswith('Transformer / ')]
    assert row(before) and row(before) == row(after), 'Changed exploratory numerical row'
    assert r'\multicolumn{3}{c}{Internal validation}' in after
    assert r'\multicolumn{3}{c}{External test}' in after
    assert after.count(r'\multicolumn{3}') == 2
