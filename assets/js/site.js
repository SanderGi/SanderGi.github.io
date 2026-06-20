function isInsideLink(element) {
  for (let node = element; node && node !== document.body; node = node.parentElement) {
    if (node.tagName === "A") return true;
  }
  return false;
}

class DetailsAccordion {
  constructor(details) {
    this.details = details;
    this.summary = details.querySelector("summary");
    this.animation = null;
    this.isClosing = false;
    this.isExpanding = false;

    const contentNodes = Array.from(details.children).filter(
      (child) => child.tagName !== "SUMMARY"
    );
    const content = document.createElement("div");
    content.className = "details-content";
    contentNodes.forEach((node) => content.appendChild(node));
    details.appendChild(content);
    this.content = content;

    this.summary.addEventListener("click", (event) => this.onClick(event));
  }

  onClick(event) {
    if (event.target !== this.summary && event.target.dataset.details !== "toggle") {
      if (!isInsideLink(event.target)) event.preventDefault();
      return;
    }

    event.preventDefault();
    this.details.style.overflow = "hidden";

    if (this.isClosing || !this.details.open) {
      this.open();
    } else {
      this.shrink();
    }
  }

  shrink() {
    this.isClosing = true;
    const startHeight = `${this.details.offsetHeight}px`;
    const endHeight = `${this.summary.offsetHeight}px`;
    this.animation?.cancel();
    this.animation = this.details.animate(
      { height: [startHeight, endHeight] },
      { duration: 260, easing: "ease-out" }
    );
    this.animation.onfinish = () => this.finish(false);
    this.animation.oncancel = () => (this.isClosing = false);
  }

  open() {
    this.details.style.height = `${this.details.offsetHeight}px`;
    this.details.open = true;
    window.requestAnimationFrame(() => this.expand());
  }

  expand() {
    this.isExpanding = true;
    const startHeight = `${this.details.offsetHeight}px`;
    const endHeight = `${this.summary.offsetHeight + this.content.offsetHeight}px`;
    this.animation?.cancel();
    this.animation = this.details.animate(
      { height: [startHeight, endHeight] },
      { duration: 260, easing: "ease-out" }
    );
    this.animation.onfinish = () => this.finish(true);
    this.animation.oncancel = () => (this.isExpanding = false);
  }

  finish(open) {
    this.details.open = open;
    this.animation = null;
    this.isClosing = false;
    this.isExpanding = false;
    this.details.style.height = "";
    this.details.style.overflow = "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("details").forEach((details) => new DetailsAccordion(details));

  const navLinks = Array.from(document.querySelectorAll("#nav a"));
  if (window.location.pathname.endsWith("blog.html")) {
    const writing = navLinks.find((link) => /blog\.html$/.test(link.getAttribute("href") || ""));
    if (writing) writing.classList.add("active");
    return;
  }

  const localSectionLinks = navLinks.filter((link) => {
    const href = link.getAttribute("href") || "";
    return href.startsWith("#") || href.startsWith("index.html#");
  });
  const sections = localSectionLinks
    .map((link) => {
      const id = (link.getAttribute("href").split("#")[1] || "").trim();
      return { link, section: id ? document.getElementById(id) : null };
    })
    .filter((item) => item.section);

  function setActive(link) {
    navLinks.forEach((item) => item.classList.toggle("active", item === link));
  }

  function updateActiveSection() {
    if (!sections.length) {
      const writing = navLinks.find((link) => /blog\.html$/.test(link.getAttribute("href") || ""));
      if (writing) setActive(writing);
      return;
    }

    const offset = window.scrollY + document.getElementById("nav").offsetHeight + 24;
    const current = sections.reduce((best, item) => {
      const top = item.section.offsetTop;
      return top <= offset && top > best.top ? { ...item, top } : best;
    }, { link: sections[0].link, top: -Infinity });
    setActive(current.link);
  }

  updateActiveSection();
  window.addEventListener("scroll", updateActiveSection, { passive: true });
});
