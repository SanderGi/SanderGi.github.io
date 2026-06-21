(() => {
  const root = document.getElementById("ekichabi-demo");
  if (!root) return;

  const shortCode = "*149*46#";
  const screen = root.querySelector("[data-ussd-screen]");
  const inputLine = root.querySelector("[data-ussd-input]");
  const languageButton = root.querySelector("[data-ussd-language]");

  const text = {
    en: {
      switchLanguage: "Switch to Swahili",
      off: "eKichabi demo\nDial *149*46#\nand press Call.",
      wrongCode: "Shortcode not recognized.\nDial *149*46#.",
      home: "Welcome to eKichabi!\nSelect an option:",
      category: "Search by category",
      location: "Browse by location",
      search: "Search",
      instructions: "Instructions",
      selectCategory: "Select a category:",
      selectLocation: "Select a location:",
      selectBusiness: "Select a Business:",
      searchFor: "What would you like to search for?",
      businessName: "Business name",
      product: "Product/Service",
      ownerName: "Owner name",
      help: "Enter 99 to go back from any screen.\nEnter 100 for home.\nPress 109 to end the session.",
      invalid: "Your input is not a valid menu item. Please try again.",
      ended: "Your eKichabi session has ended.\nDial *149*46# to start again.",
      allBusinesses: (count) => `All businesses (${count})`,
      back: "99. Back",
      end: "109. End",
      next: "next",
      nextBack: "0. next, 99. back",
      nextOnly: "0. next",
      owner: "Owner",
      located: "Located in",
      contact: "Contact",
      input: "Input",
      empty: "No matching demo businesses were found."
    },
    sw: {
      switchLanguage: "Switch to English",
      off: "Onyesho la eKichabi\nPiga *149*46#\nkisha bonyeza Call.",
      wrongCode: "Msimbo haujatambuliwa.\nPiga *149*46#.",
      home: "Karibu eKichabi!\nChagua ya ziada:",
      category: "Tafuta kwa kuchagua sekta",
      location: "Tafuta kwa kuchagua mahali",
      search: "Tafuta kwa kuandika",
      instructions: "Maelekezo",
      selectCategory: "Chagua sekta:",
      selectLocation: "Chagua mahali:",
      selectBusiness: "Chagua biashara:",
      searchFor: "Ungependa kutafuta nini?",
      businessName: "Jina la biashara",
      product: "Bidhaa/Huduma",
      ownerName: "Jina la mmiliki",
      help: "Ingiza 99 ili kurudi nyuma.\nWeka 100 kwa skrini ya kwanza.\nBonyeza 109 kumaliza.",
      invalid: "Ingizo lako si chaguo sahihi. Jaribu tena.",
      ended: "Kikao chako cha eKichabi kimeisha.\nPiga *149*46# kuanza tena.",
      allBusinesses: (count) => `Biashara zote (${count})`,
      back: "99. Rudi nyuma",
      end: "109. Maliza",
      next: "Mbele",
      nextBack: "0. Mbele, 99. Rudi nyuma",
      nextOnly: "0. Mbele",
      owner: "Mmiliki",
      located: "Wapi",
      contact: "Mawasiliano",
      input: "Ingizo",
      empty: "Hakuna biashara za mfano zilizopatikana."
    }
  };

  const businesses = [
    ["Pokémon Center", "Nurse Joy", "Health", "Afya", "Kanto", "Kanto", "potions, berries, clinic", "dawa, matunda, kliniki", "555-0101"],
    ["Ollivanders", "Garrick Ollivander", "Skilled Trades", "Fundi", "Diagon Alley", "Diagon Alley", "wands, repairs, fittings", "fimbo, ukarabati, vipimo", "555-0102"],
    ["The Shop of Curiosities", "Mr. Magorium", "Merchant/retail", "Rejareja", "Wonder Avenue", "Wonder Avenue", "toys, oddities, gifts", "vichezeo, vitu adimu, zawadi", "555-0103"],
    ["Willy Wonka's Chocolate Factory", "W. Wonka", "Agricultural processing", "Usindikaji", "Sweetwater", "Sweetwater", "cocoa, candy, tours", "kakao, pipi, ziara", "555-0104"],
    ["The Shop Around the Corner", "Kathleen Kelly", "Merchant/retail", "Rejareja", "Upper West Side", "Upper West Side", "books, stationery", "vitabu, vifaa vya kuandika", "555-0105"],
    ["The Travel Book Company", "William Thacker", "Merchant/retail", "Rejareja", "Notting Hill", "Notting Hill", "maps, travel books", "ramani, vitabu vya safari", "555-0106"],
    ["Al's Toy Barn", "Al McWhiggin", "Merchant/retail", "Rejareja", "Tri-County", "Tri-County", "toys, batteries, repairs", "vichezeo, betri, ukarabati", "555-0107"],
    ["Women & Women First", "Toni and Candace", "Services", "Huduma", "Portland", "Portland", "books, workshops", "vitabu, mafunzo", "555-0108"],
    ["Ray's Occult Books", "Ray Stantz", "Merchant/retail", "Rejareja", "New York", "New York", "books, paranormal supplies", "vitabu, vifaa vya ajabu", "555-0109"],
    ["The Android's Dungeon", "Comic Book Guy", "Merchant/retail", "Rejareja", "Springfield", "Springfield", "comics, cards, collectibles", "katuni, kadi, makusanyo", "555-0110"],
    ["The Floating Candle Cafe", "H. Hufflepuff", "Food", "Chakula", "Hogwarts", "Hogwarts", "tea, cakes, butterbeer", "chai, keki, butterbeer", "555-0111"],
    ["The Last Homely Salon", "Elrond", "Services", "Huduma", "Rivendell", "Rivendell", "hair, herbs, harp music", "nywele, mitishamba, muziki", "555-0112"],
    ["The Green Lens Optometry", "O. Gale", "Health", "Afya", "Emerald City", "Emerald City", "glasses, eye exams", "miwani, vipimo vya macho", "555-0113"],
    ["The Crypt & Kettle", "Old Nan", "Food", "Chakula", "Winterfell", "Winterfell", "stew, tea, warm fires", "mchuzi, chai, moto", "555-0114"],
    ["Gargoyle Restorations", "Bruce W.", "Repairs", "Fundi", "Gotham City", "Gotham City", "stonework, roof repairs", "mawe, ukarabati wa paa", "555-0115"],
    ["Vibranium Glow Tech", "Shuri", "Skilled Trades", "Fundi", "Wakanda", "Wakanda", "solar, sensors, gadgets", "jua, vihisi, vifaa", "555-0116"],
    ["The Energon Fuel Bar", "Bumblebee", "Transport", "Usafirishaji", "Cybertron", "Cybertron", "fuel, parts, charging", "mafuta, vipuri, kuchaji", "555-0117"],
    ["Scrap & Scum Mechanics", "Peli Motto", "Repairs", "Fundi", "Mos Eisley", "Mos Eisley", "droids, scrap, engines", "roboti, vyuma, injini", "555-0118"],
    ["The Second Breakfast Bakery", "Samwise Gamgee", "Food", "Chakula", "The Shire", "The Shire", "bread, jam, potatoes", "mkate, jamu, viazi", "555-0119"],
    ["The Town Meeting Diner", "Luke Danes", "Food", "Chakula", "Stars Hollow", "Stars Hollow", "coffee, pancakes, gossip", "kahawa, chapati, habari", "555-0120"],
    ["The Krusty Krab", "Eugene Krabs", "Food", "Chakula", "Bikini Bottom", "Bikini Bottom", "burgers, fries, sea snacks", "baga, chipsi, vitafunwa", "555-0121"],
    ["The Merry Archer Gear", "Robin Hood", "Merchant/retail", "Rejareja", "Sherwood Forest", "Sherwood Forest", "bows, boots, repairs", "pinde, buti, ukarabati", "555-0122"],
    ["The Maze Hedge Trimmers", "Wendy Torrance", "Services", "Huduma", "Sidewinder", "Sidewinder", "hedges, snow shovels", "ua, koleo la theluji", "555-0123"],
    ["Owl Post Express", "Minerva McGonagall", "Transport", "Usafirishaji", "Hogwarts", "Hogwarts", "mail, parcels, owls", "barua, mizigo, bundi", "555-0124"],
    ["The Daily Planet", "Perry White", "Services", "Huduma", "Metropolis", "Metropolis", "news, printing, ads", "habari, uchapishaji, matangazo", "555-0125"],
    ["Slay Speech", "Koel Labs", "Education", "Elimu", "App", "App", "language learning, pronunciation", "kujifunza lugha, matamshi", "slayspeech.com"]
  ].map(([name, owner, category, categorySw, location, locationSw, keywords, keywordsSw, contact]) => ({
    name,
    owner,
    category,
    categorySw,
    location,
    locationSw,
    keywords,
    keywordsSw,
    contact
  }));

  const state = {
    active: false,
    input: "",
    lang: "en",
    screen: { type: "dial" },
    stack: [],
    notice: ""
  };

  function t(key) {
    return text[state.lang][key];
  }

  function labelFor(kind, value) {
    if (state.lang === "en") return value;
    const match = businesses.find((business) => business[kind] === value);
    if (!match) return value;
    return kind === "category" ? match.categorySw : match.locationSw;
  }

  function unique(values) {
    return [...new Set(values)].sort((a, b) => a.localeCompare(b));
  }

  function go(next, remember = true) {
    if (remember && state.active && state.screen.type !== "dial") {
      state.stack.push(state.screen);
    }
    state.screen = next;
    state.input = "";
    state.notice = "";
    render();
  }

  function home() {
    state.active = true;
    state.stack = [];
    go({ type: "home" }, false);
  }

  function endSession() {
    state.active = false;
    state.stack = [];
    state.input = "";
    state.screen = { type: "ended" };
    render();
  }

  function back() {
    state.screen = state.stack.pop() || { type: "home" };
    state.input = "";
    state.notice = "";
    render();
  }

  function pageItems(items, page, size = 6) {
    const start = page * size;
    return {
      visible: items.slice(start, start + size),
      hasNext: start + size < items.length
    };
  }

  function renderMenu(title, entries, footer = t("back")) {
    return [
      state.notice,
      title,
      ...entries.map((entry) => `${entry.key}. ${entry.label}`),
      footer
    ].filter(Boolean).join("\n");
  }

  function renderList(title, items, page = 0, size = 6) {
    const { visible, hasNext } = pageItems(items, page, size);
    const entries = visible.map((item, index) => ({
      key: String(index + 1),
      label: item.label
    }));
    if (hasNext) entries.push({ key: "0", label: t("next") });
    return renderMenu(title, entries);
  }

  function businessDetail(business, index, list) {
    const keywords = state.lang === "en" ? business.keywords : business.keywordsSw;
    const location = state.lang === "en" ? business.location : business.locationSw;
    const footer = index < list.length - 1 ? t("nextBack") : t("back");
    return [
      state.notice,
      business.name,
      "--",
      keywords,
      `${t("owner")}: ${business.owner}`,
      `${t("located")}: ${location}`,
      `${t("contact")}: ${business.contact}`,
      footer
    ].filter(Boolean).join("\n");
  }

  function visibleBusinesses(list, page) {
    return pageItems(list, page, 5);
  }

  function currentText() {
    const current = state.screen;
    if (current.type === "dial") return state.notice || t("off");
    if (current.type === "ended") return t("ended");
    if (current.type === "home") {
      return renderMenu(t("home"), [
        { key: "1", label: t("category") },
        { key: "2", label: t("location") },
        { key: "3", label: t("search") },
        { key: "4", label: t("instructions") }
      ], t("end"));
    }
    if (current.type === "help") return `${state.notice ? `${state.notice}\n` : ""}${t("help")}\n${t("back")}`;
    if (current.type === "categories") {
      const categories = unique(businesses.map((business) => business.category)).map((category) => ({
        value: category,
        label: labelFor("category", category)
      }));
      return renderList(t("selectCategory"), categories, current.page);
    }
    if (current.type === "locations") {
      const locations = unique(businesses.map((business) => business.location)).map((location) => ({
        value: location,
        label: labelFor("location", location)
      }));
      return renderList(t("selectLocation"), locations, current.page);
    }
    if (current.type === "search") {
      return renderMenu(t("searchFor"), [
        { key: "1", label: t("businessName") },
        { key: "2", label: t("location") },
        { key: "3", label: t("product") },
        { key: "4", label: t("ownerName") }
      ]);
    }
    if (current.type === "businesses") {
      const { visible, hasNext } = visibleBusinesses(current.items, current.page);
      const entries = visible.map((business, index) => ({
        key: String(index + 1),
        label: business.name
      }));
      if (hasNext) entries.push({ key: "0", label: t("next") });
      const title = current.title || t("selectBusiness");
      return renderMenu(title, entries);
    }
    if (current.type === "detail") return businessDetail(current.business, current.index, current.items);
    return t("invalid");
  }

  function render() {
    screen.textContent = currentText();
    inputLine.textContent = state.input ? `${t("input")}: ${state.input}` : "";
    languageButton.textContent = t("switchLanguage");
  }

  function chooseFromPaged(values, page, input, onSelect) {
    const { visible, hasNext } = pageItems(values, page, 6);
    if (input === "0" && hasNext) return { nextPage: true };
    const index = Number(input) - 1;
    if (index >= 0 && index < visible.length) onSelect(visible[index]);
    else invalid();
    return {};
  }

  function chooseBusiness(current, input) {
    const { visible, hasNext } = visibleBusinesses(current.items, current.page);
    if (input === "0" && hasNext) {
      go({ ...current, page: current.page + 1 });
      return;
    }
    const index = Number(input) - 1;
    if (index >= 0 && index < visible.length) {
      const absoluteIndex = current.page * 5 + index;
      go({ type: "detail", business: visible[index], index: absoluteIndex, items: current.items });
    } else {
      invalid();
    }
  }

  function invalid() {
    state.notice = t("invalid");
    state.input = "";
    render();
  }

  function submitActive(input) {
    if (input === "99") return back();
    if (input === "100") return home();
    if (input === "109") return endSession();

    const current = state.screen;
    if (current.type === "home") {
      if (input === "1") return go({ type: "categories", page: 0 });
      if (input === "2") return go({ type: "locations", page: 0 });
      if (input === "3") return go({ type: "search" });
      if (input === "4") return go({ type: "help" });
    }
    if (current.type === "help") return invalid();
    if (current.type === "search") {
      if (input === "1") return go({ type: "businesses", title: t("selectBusiness"), items: businesses, page: 0 });
      if (input === "2") return go({ type: "locations", page: 0 });
      if (input === "3") return go({ type: "categories", page: 0 });
      if (input === "4") {
        const owners = [...businesses].sort((a, b) => a.owner.localeCompare(b.owner));
        return go({ type: "businesses", title: t("ownerName"), items: owners, page: 0 });
      }
    }
    if (current.type === "categories") {
      const categories = unique(businesses.map((business) => business.category)).map((category) => ({
        value: category,
        label: labelFor("category", category)
      }));
      const result = chooseFromPaged(categories, current.page, input, (category) => {
        const items = businesses.filter((business) => business.category === category.value);
        go({ type: "businesses", title: category.label, items, page: 0 });
      });
      if (result.nextPage) go({ ...current, page: current.page + 1 });
      return;
    }
    if (current.type === "locations") {
      const locations = unique(businesses.map((business) => business.location)).map((location) => ({
        value: location,
        label: labelFor("location", location)
      }));
      const result = chooseFromPaged(locations, current.page, input, (location) => {
        const items = businesses.filter((business) => business.location === location.value);
        go({ type: "businesses", title: location.label, items, page: 0 });
      });
      if (result.nextPage) go({ ...current, page: current.page + 1 });
      return;
    }
    if (current.type === "businesses") return chooseBusiness(current, input);
    if (current.type === "detail") {
      if (input === "0" && current.index < current.items.length - 1) {
        const nextIndex = current.index + 1;
        return go({
          type: "detail",
          business: current.items[nextIndex],
          index: nextIndex,
          items: current.items
        }, false);
      }
    }
    invalid();
  }

  function submit() {
    const input = state.input.trim();
    if (!state.active) {
      if (input === shortCode) home();
      else {
        state.notice = t("wrongCode");
        state.input = "";
        render();
      }
      return;
    }
    if (!input) return;
    submitActive(input);
  }

  root.querySelectorAll("[data-ussd-key]").forEach((button) => {
    button.addEventListener("click", () => {
      state.input += button.dataset.ussdKey;
      render();
    });
  });

  root.querySelector("[data-ussd-clear]").addEventListener("click", () => {
    state.input = state.input.slice(0, -1);
    render();
  });

  root.querySelector("[data-ussd-call]").addEventListener("click", submit);

  languageButton.addEventListener("click", () => {
    state.lang = state.lang === "en" ? "sw" : "en";
    state.notice = "";
    render();
  });

  root.addEventListener("keydown", (event) => {
    if (!/^[0-9*#]$/.test(event.key)) return;
    state.input += event.key;
    render();
  });

  render();
})();
