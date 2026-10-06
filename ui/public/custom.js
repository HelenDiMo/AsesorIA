/*
 * Asesor Fiscal IA — targeted interface fixes.
 * 1) Custom editor placeholder with a tax hint.
 * 2) Localization of two strings Chainlit leaves in English
 *    (theme button accessible label and task-stop notice).
 * Defensive: if the selector or the string does not exist, it does nothing.
 */
(function () {
  var PLACEHOLDER = "Pregunta sobre tus obligaciones, gastos, IVA o IRPF…";
  var DEFAULT_MARK = "Escribe tu mensaje";
  var THEME_LABEL = ["Toggle theme", "Cambiar tema"];
  var STOP_NOTICE = [
    "Task manually stopped.",
    "Consulta detenida. Puedes volver a preguntar cuando quieras.",
  ];

  function applyPlaceholder() {
    var nodes = document.querySelectorAll("textarea");
    for (var i = 0; i < nodes.length; i++) {
      var ta = nodes[i];
      var current = ta.getAttribute("placeholder") || "";
      if (current.indexOf(DEFAULT_MARK) !== -1 || current === "") {
        if (ta.getAttribute("placeholder") !== PLACEHOLDER) {
          ta.setAttribute("placeholder", PLACEHOLDER);
        }
      }
    }
  }

  function localizeThemeLabel() {
    var spans = document.querySelectorAll(".sr-only");
    for (var i = 0; i < spans.length; i++) {
      if (spans[i].textContent.trim() === THEME_LABEL[0]) {
        spans[i].textContent = THEME_LABEL[1];
      }
    }
  }

  function localizeStopNotice() {
    if (!document.body) return;
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    var node;
    while ((node = walker.nextNode())) {
      if (node.data && node.data.trim() === STOP_NOTICE[0]) {
        node.data = " " + STOP_NOTICE[1] + " ";
      }
    }
  }

  function apply() {
    applyPlaceholder();
    localizeThemeLabel();
    localizeStopNotice();
  }

  apply();

  if (window.MutationObserver) {
    new MutationObserver(apply).observe(document.body, {
      childList: true,
      subtree: true,
    });
  }
})();
