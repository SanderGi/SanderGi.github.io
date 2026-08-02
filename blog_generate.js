// node --watch blog_generate.js

const fs = require("fs");

function read_image_dimensions(url) {
  if (!url || /^(?:https?:|data:)/i.test(url)) return null;
  const filename = url.startsWith("/") ? `.${url}` : url;

  try {
    const data = fs.readFileSync(filename);

    if (data.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"))) {
      return { width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
    }

    if (data.toString("ascii", 0, 4) === "RIFF" && data.toString("ascii", 8, 12) === "WEBP") {
      const chunk = data.toString("ascii", 12, 16);
      if (chunk === "VP8X") {
        return {
          width: 1 + data.readUIntLE(24, 3),
          height: 1 + data.readUIntLE(27, 3),
        };
      }
      if (chunk === "VP8 ") {
        return {
          width: data.readUInt16LE(26) & 0x3fff,
          height: data.readUInt16LE(28) & 0x3fff,
        };
      }
      if (chunk === "VP8L") {
        const bits = data.readUInt32LE(21);
        return {
          width: 1 + (bits & 0x3fff),
          height: 1 + ((bits >> 14) & 0x3fff),
        };
      }
    }

    if (data[0] === 0xff && data[1] === 0xd8) {
      const startOfFrame = new Set([
        0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
        0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
      ]);
      let offset = 2;
      while (offset + 8 < data.length) {
        if (data[offset] !== 0xff) {
          offset += 1;
          continue;
        }
        const marker = data[offset + 1];
        if (startOfFrame.has(marker)) {
          return {
            width: data.readUInt16BE(offset + 7),
            height: data.readUInt16BE(offset + 5),
          };
        }
        if (marker === 0xd8 || marker === 0xd9) {
          offset += 2;
          continue;
        }
        offset += 2 + data.readUInt16BE(offset + 2);
      }
    }
  } catch (error) {
    console.warn(`Could not read dimensions for ${url}: ${error.message}`);
  }

  return null;
}

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

function read_meta(content, name, fallback = "") {
  const match = content.match(
    new RegExp(`<meta\\s+name=["']${name}["']\\s+content=["']([^"']*)["']`, "i")
  );
  return match ? match[1] : fallback;
}

function read_property(content, property, fallback = "") {
  const match = content.match(
    new RegExp(`<meta\\s+property=["']${property}["']\\s+content=["']([^"']*)["']`, "i")
  );
  return match ? match[1] : fallback;
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
      if (!file.endsWith(".html")) return;

      const url = "./blog/" + file;
      const content = fs.readFileSync(url).toString();
      if (read_meta(content, "draft").toLowerCase() === "true") return;

      const title = content.split("<title>", 2)[1].split("</", 2)[0];
      const author = read_meta(content, "author", "Alexander Metzger");
      const description = read_meta(content, "description");
      const keywords = read_meta(content, "keywords", "Research")
        .split(",");
      const created = read_meta(content, "dcterms.created");
      const [image, alt] = read_property(
        content,
        "og:image",
        "/images/profile.jpg | Alex Metzger"
      ).split(" | ");
      const dimensions = read_image_dimensions(image);
      const dimension_attributes = dimensions
        ? ` width="${dimensions.width}" height="${dimensions.height}"`
        : "";
      blog_list.push([
        /* html */ `
          <article class="card blog-card" data-tags="${keywords}">
            <a
              href="${url}"
              class="card-img">
              <img loading="lazy" decoding="async"${dimension_attributes} src="${image}" alt="${alt}"/>
            </a>
            <div class="card-content">
              <h3><a href="${url}">${title}</a></h3>
              <p>${markdown_to_html(description)}</p>
              <div class="blog-meta">
                <span>${author}</span>
                <span>${created}</span>
              </div>
              <div class="pills">
                ${keywords.map((k) => "<span>" + k.trim() + "</span> ").join("")}
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

update_blog_list();

if (process.argv.includes("--watch")) {
  fs.watch(".", (_, filename) => {
    if (!["blog_template.html"].includes(filename)) return;
    update_blog_list();
  });

  fs.watch("./blog", () => {
    update_blog_list();
  });
}
