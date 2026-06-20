// node --watch blog_generate.js

const fs = require("fs");

function markdown_to_html(markdown) {
  return markdown
    .replace(/^### (.*$)/gim, "<h3>$1</h3>") // h3 tag
    .replace(/^## (.*$)/gim, "<h2>$1</h2>") // h2 tag
    .replace(/^# (.*$)/gim, "<h1>$1</h1>") // h1 tag
    .replace(/\*\*(.*)\*\*/gim, "<b>$1</b>") // bold text
    .replace(/\*(.*)\*/gim, "<i>$1</i>") // italic text
    .replace(/\r\n|\r|\n/gim, "<br>") // linebreaks
    .replace(/\[([^\[]+)\](\(([^)]*))\)/gim, '<a href="$3">$1</a>'); // anchor tags
}

function update_blog_list() {
  const template = fs.readFileSync("./blog_template.html").toString();
  const [template_start, template_end] = template.split("{blog_list}", 2);

  const blog_list = [];
  fs.readdir("./blog", (err, files) => {
    if (err) {
      console.error("Error reading directory:", err);
      return;
    }

    files.forEach((file) => {
      const url = "./blog/" + file;
      const content = fs.readFileSync(url).toString();
      const title = content.split("<title>", 2)[1].split("</", 2)[0];
      const author = content
        .split('<meta name="author" content="', 2)[1]
        .split('"', 2)[0];
      const description = content
        .split('<meta name="description" content="', 2)[1]
        .split('"', 2)[0];
      const keywords = content
        .split('<meta name="keywords" content="', 2)[1]
        .split('"', 2)[0]
        .split(",");
      const created = content
        .split('<meta name="dcterms.created" content="', 2)[1]
        .split('"', 2)[0];
      const [image, alt] = content
        .split('<meta property="og:image" content="', 2)[1]
        .split('"', 2)[0]
        .split(" | ");
      blog_list.push([
        /* html */ `
          <article class="card" data-tags="${keywords}">
            <a
              href="${url}"
              class="card-img">
              <img src="${image}" alt="${alt}"/>
            </a>
            <div class="card-content">
              <h3><a href="${url}">${title}</a></h3>
              <p>${markdown_to_html(description)}</p>
              <div class="pills">
                ${keywords.map((k) => "<span>" + k + "</span> ").join("")}
                <div style="float: right; margin-right: 1em">
                  ${author} • ${created}
                </div>
              </div>
            </div>
          </article>
        `,
        created,
      ]);
    });

    fs.writeFileSync(
      "blog.html",
      [
        template_start,
        ...blog_list.sort((a, b) => (a[1] < b[1] ? 1 : -1)).map((r) => r[0]),
        template_end,
      ].join("")
    );
  });
}

fs.watch(".", (_, filename) => {
  if (!["blog_template.html"].includes(filename)) return;
  update_blog_list();
});

fs.watch("./blog", () => {
  update_blog_list();
});

update_blog_list();
