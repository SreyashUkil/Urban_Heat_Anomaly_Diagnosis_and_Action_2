"""
Make the generated map self-contained.

Folium normally loads Leaflet and its plugins from public CDNs (cdn.jsdelivr.net,
cdnjs, code.jquery.com). Some networks/ISPs block cdn.jsdelivr.net, which leaves the
page blank except for the custom panels. This module replaces every CDN <script>/<link>
tag with the same library embedded inline from ./vendor, and embeds the images/fonts
those stylesheets reference as data URIs. After this, only the background map tiles
need the internet.
"""
import base64
import mimetypes
import os
import re

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
MIME = {".png": "image/png", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".woff": "font/woff",
        ".ttf": "font/ttf", ".gif": "image/gif", ".jpg": "image/jpeg"}
DROP = ("bootstrap-glyphicons.css",)   # unused by our map, no local copy needed


def _local(url):
    p = os.path.join(VENDOR, *url.split("://", 1)[1].split("?")[0].split("/"))
    return p if os.path.isfile(p) else None


def _embed_css_urls(css, css_path):
    base = os.path.dirname(css_path)

    def rep(m):
        ref = m.group(1).strip().strip("'\"")
        if ref.startswith(("data:", "http", "#", "//")):
            return m.group(0)
        f = os.path.normpath(os.path.join(base, ref.split("?")[0].split("#")[0]))
        if not os.path.isfile(f):
            return m.group(0)
        mime = MIME.get(os.path.splitext(f)[1].lower()) or mimetypes.guess_type(f)[0] or "application/octet-stream"
        return f"url(data:{mime};base64,{base64.b64encode(open(f, 'rb').read()).decode()})"
    return re.sub(r"url\(([^)]+)\)", rep, css)


ERROR_BANNER = r"""<script>
/* Shows any JavaScript problem directly on the page instead of a silent blank map */
(function(){
  function banner(msg){
    var d=document.getElementById('uhi-err');
    if(!d){d=document.createElement('div');d.id='uhi-err';
      d.style.cssText='position:fixed;top:0;left:0;right:0;z-index:100000;background:#b71c1c;color:#fff;'+
        'font:14px/1.4 system-ui,Segoe UI,Arial;padding:10px 14px;white-space:pre-wrap';
      (document.body||document.documentElement).appendChild(d);}
    d.textContent+=(d.textContent?'\n':'')+'⚠ '+msg;
  }
  window.addEventListener('error',function(e){banner('Error: '+e.message+(e.filename?' ('+e.filename.split('/').pop()+':'+e.lineno+')':''));});
  window.addEventListener('load',function(){
    if(typeof L==='undefined') banner('Map library (Leaflet) did not load. This HTML was built without the bundled libraries – re-run main.py from the NEW project folder (it must contain the vendor/ folder).');
  });
})();
</script>
"""


def inline_assets(html_path):
    html = open(html_path, encoding="utf-8").read()
    missing = []

    def script(m):
        url = m.group(1)
        p = _local(url)
        if not p:
            missing.append(url); return m.group(0)
        js = open(p, encoding="utf-8").read().replace("</script", "<\\/script")
        return f"<script>/* {url.split('/')[-1]} (bundled) */\n{js}\n</script>"

    def link(m):
        url = m.group(1)
        if url.endswith(DROP):
            return ""
        p = _local(url)
        if not p:
            missing.append(url); return m.group(0)
        css = _embed_css_urls(open(p, encoding="utf-8").read(), p)
        return f"<style>/* {url.split('/')[-1]} (bundled) */\n{css}\n</style>"

    if not os.path.isdir(VENDOR):
        print("\n  ❌ vendor/ folder not found next to main.py – the map libraries cannot be bundled.\n"
              "     Extract the whole zip again (including the vendor folder) and run from that folder.\n")
    html = re.sub(r'<script src="(https?://[^"]+)"></script>', script, html)
    html = re.sub(r'<link rel="stylesheet" href="(https?://[^"]+)"\s*/?>', link, html)
    html = html.replace("<head>", "<head>\n" + ERROR_BANNER, 1)
    open(html_path, "w", encoding="utf-8").write(html)
    n = html.count("(bundled) */")
    print(f"  ✓ map libraries bundled into the HTML: {n} files" + ("" if not missing else f", {len(missing)} still online"))
    if missing:
        print("  note: these libraries still load from the internet:", *missing, sep="\n    ")
    return missing
