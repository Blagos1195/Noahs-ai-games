const boardElement = document.querySelector("#board");
const columnButtons = [...document.querySelectorAll(".column-button")];
const modeButtons = [...document.querySelectorAll(".mode-button")];
const statusHeading = document.querySelector("#game-status");
const roundLabel = document.querySelector("#round-label");
const turnPill = document.querySelector("#turn-pill");
const turnDisc = turnPill.querySelector("span");
const turnLabel = document.querySelector("#turn-label");
const modelStatus = document.querySelector("#model-status");
const nextTitle = document.querySelector("#next-title");
const nextCopy = document.querySelector("#next-copy");
const scoreX = document.querySelector("#score-x");
const scoreO = document.querySelector("#score-o");
const scoreDraw = document.querySelector("#score-draw");
const scoreXName = document.querySelector("#score-x-name");
const scoreOName = document.querySelector("#score-o-name");
const opponentName = document.querySelector("#opponent-name");

let mode = "ai";
let currentPlayer = "X";
let aiPlayer = "O";
let gameOver = false;
let aiBusy = false;
let engineReady = false;
let pyodide = null;
let aiLoading = null;
let gameVersion = 0;
let board = Array(42).fill("");
let winningLine = [];
const scores = { X: 0, O: 0, draw: 0 };

function otherPlayer(player) {
  return player === "X" ? "O" : "X";
}

function setModelStatus(message, ready = false) {
  modelStatus.textContent = message;
  modelStatus.parentElement.querySelector(".status-dot").style.background = ready ? "#83b99c" : "#999";
}

function loadAI() {
  if (engineReady) return Promise.resolve();
  if (aiLoading) return aiLoading;

  aiLoading = (async () => {
    setModelStatus("Loading Python runtime...");
    const [runtime, response] = await Promise.all([
      loadPyodide(),
      fetch("connect4_ntuple.py?v=60c319a")
    ]);
    if (!response.ok) throw new Error("Could not load connect4_ntuple.py");
    const pythonSource = await response.text();
    pyodide = runtime;
    await pyodide.loadPackage("numpy");
    pyodide.FS.writeFile("connect4_ntuple.py", pythonSource);
    await pyodide.runPythonAsync(
      "import connect4_ntuple as c4\nbrowser_game = c4.ConnectFour()\nbrowser_agent = c4.NTupleAgent()"
    );

    let modelLoaded = false;
    try {
      const modelResponse = await fetch("models/connect4_ntuple.npz");
      if (modelResponse.ok) {
        const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());
        pyodide.FS.writeFile("connect4_ntuple.npz", modelBytes);
        await pyodide.runPythonAsync("browser_agent.load('connect4_ntuple.npz')");
        modelLoaded = true;
      }
    } catch (error) {
      console.warn("No trained checkpoint loaded.", error);
    }

    engineReady = true;
    setModelStatus(modelLoaded ? "Python AI ready · trained model loaded" : "Python AI ready · no trained model", true);
  })().catch(error => {
    console.error(error);
    setModelStatus("Python AI failed to load");
    aiLoading = null;
    throw error;
  });

  return aiLoading;
}

function pythonBoard() {
  const proxy = pyodide.runPython("c4.board_cells(browser_game)");
  const relative = proxy.toJs();
  proxy.destroy();
  return relative.flat().map(value => value === "current" ? currentPlayer : value === "opponent" ? otherPlayer(currentPlayer) : "");
}

function findWinningLine(position, player) {
  const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 7; col++) {
      if (position[row * 7 + col] !== player) continue;
      for (const [dr, dc] of directions) {
        const line = [];
        for (let step = 0; step < 4; step++) {
          const r = row + dr * step;
          const c = col + dc * step;
          if (r < 0 || r >= 6 || c < 0 || c >= 7 || position[r * 7 + c] !== player) break;
          line.push(r * 7 + c);
        }
        if (line.length === 4) return line;
      }
    }
  }
  return [];
}

function render() {
  if (engineReady) board = pythonBoard();
  boardElement.replaceChildren();
  board.forEach((value, index) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (value) cell.classList.add(value === "X" ? "x" : "o");
    if (winningLine.includes(index)) cell.classList.add("winning");
    cell.setAttribute("role", "gridcell");
    const row = Math.floor(index / 7) + 1;
    const col = index % 7 + 1;
    cell.setAttribute("aria-label", `Row ${row}, column ${col}, ${value || "empty"}`);
    boardElement.append(cell);
  });

  columnButtons.forEach((button, col) => {
    const full = engineReady && board.filter((_, index) => index % 7 === col).every(Boolean);
    button.disabled = !engineReady || gameOver || aiBusy || full || (mode === "ai" && currentPlayer === aiPlayer);
  });

  scoreX.textContent = scores.X;
  scoreO.textContent = scores.O;
  scoreDraw.textContent = scores.draw;
  turnPill.style.visibility = gameOver ? "hidden" : "visible";
  turnDisc.className = currentPlayer === "X" ? "x-disc" : "o-disc";
  turnLabel.textContent = mode === "ai"
    ? currentPlayer === "X" ? "YOUR MOVE" : "AI THINKING"
    : `PLAYER ${currentPlayer === "X" ? "1" : "2"}`;
}

function updateModeLabels() {
  const aiMode = mode === "ai";
  scoreXName.textContent = aiMode ? aiPlayer === "X" ? "AI" : "YOU" : "PLAYER 1";
  scoreOName.textContent = aiMode ? aiPlayer === "O" ? "AI" : "YOU" : "PLAYER 2";
  opponentName.textContent = aiMode ? "AI" : "Player 2";
  nextTitle.textContent = aiMode ? currentPlayer === aiPlayer ? "AI turn" : "Your move" : `Player ${currentPlayer === "X" ? "1" : "2"} starts`;
  nextCopy.textContent = aiMode ? `You play ${aiPlayer === "X" ? "yellow" : "red"}. The AI plays ${aiPlayer === "X" ? "red" : "yellow"}.` : "Take turns dropping counters into the board.";
}

function updateTurnText() {
  if (gameOver) return;
  if (mode === "ai") {
    roundLabel.textContent = currentPlayer === aiPlayer ? "THE AI IS THINKING" : "YOUR TURN";
    statusHeading.textContent = currentPlayer === aiPlayer ? "The AI is choosing." : "Choose a column.";
    nextTitle.textContent = currentPlayer === aiPlayer ? "AI turn" : "Your move";
    nextCopy.textContent = currentPlayer === aiPlayer ? "The Python agent is selecting a column." : `You play ${currentPlayer === "X" ? "red" : "yellow"}. The AI plays ${aiPlayer === "X" ? "red" : "yellow"}.`;
  } else {
    const player = currentPlayer === "X" ? "1" : "2";
    roundLabel.textContent = `PLAYER ${player}'S TURN`;
    statusHeading.textContent = `Player ${player}, choose a column.`;
    nextTitle.textContent = `Player ${player}'s move`;
    nextCopy.textContent = `Drop a ${currentPlayer === "X" ? "red" : "yellow"} counter.`;
  }
}

function finishGame(winner) {
  gameOver = true;
  if (winner) {
    scores[winner] += 1;
    winningLine = findWinningLine(board, winner);
    statusHeading.textContent = mode === "ai"
      ? winner === aiPlayer ? "The AI wins." : "You win."
      : `Player ${winner === "X" ? "1" : "2"} wins.`;
    roundLabel.textContent = "GAME OVER";
    nextTitle.textContent = mode === "ai"
      ? winner === aiPlayer ? "AI wins" : "You win"
      : winner === "X" ? "Player 1 wins" : "Player 2 wins";
    nextCopy.textContent = "Four in a row.";
  } else {
    scores.draw += 1;
    statusHeading.textContent = "It's a draw.";
    roundLabel.textContent = "GAME OVER";
    nextTitle.textContent = "Draw";
    nextCopy.textContent = "The board is full.";
  }
  render();
}

function playColumn(column) {
  pyodide.globals.set("browser_column", column);
  const resultProxy = pyodide.runPython("browser_game.play(browser_column)");
  const [won, draw] = resultProxy.toJs();
  resultProxy.destroy();
  const mover = currentPlayer;
  currentPlayer = otherPlayer(currentPlayer);
  board = pythonBoard();

  if (won) finishGame(mover);
  else if (draw) finishGame(null);
  else {
    updateTurnText();
    render();
  }
}

async function makeAIMove() {
  aiBusy = true;
  updateTurnText();
  render();
  try {
    await loadAI();
    const action = Number(pyodide.runPython("browser_agent.choose_action(browser_game, 0.0)"));
    if (!Number.isInteger(action) || action < 0 || action >= 7 || board[action] && board[action + 7] && board[action + 14] && board[action + 21] && board[action + 28] && board[action + 35]) {
      throw new Error("The Python agent returned an invalid move.");
    }
    aiBusy = false;
    playColumn(action);
  } catch (error) {
    console.error(error);
    aiBusy = false;
    gameOver = true;
    statusHeading.textContent = "The AI couldn't move.";
    roundLabel.textContent = "AI ERROR";
    nextTitle.textContent = "AI unavailable";
    nextCopy.textContent = error?.message || String(error);
    setModelStatus("AI error: " + (error?.message || String(error)));
    render();
  }
}

function resetPythonGame() {
  pyodide.runPython("browser_game = c4.ConnectFour()");
  board = Array(42).fill("");
  gameOver = false;
  aiBusy = false;
  winningLine = [];
  updateModeLabels();
  updateTurnText();
  render();
}

function startGame(resetScores = false) {
  const version = ++gameVersion;
  board = Array(42).fill("");
  currentPlayer = Math.random() < 0.5 ? "X" : "O";
  aiPlayer = currentPlayer === "X" ? "O" : "X";
  gameOver = false;
  aiBusy = false;
  winningLine = [];
  if (resetScores) {
    scores.X = 0;
    scores.O = 0;
    scores.draw = 0;
  }
  updateModeLabels();
  updateTurnText();
  render();
  loadAI().then(() => {
    if (version === gameVersion) {
      resetPythonGame();
      if (mode === "ai" && currentPlayer === aiPlayer) makeAIMove();
    }
  }).catch(error => {
    console.error(error);
    statusHeading.textContent = "The game engine couldn't start.";
    roundLabel.textContent = "AI ERROR";
    nextTitle.textContent = "Game setup failed";
    nextCopy.textContent = error?.message || String(error);
    setModelStatus("Setup error: " + (error?.message || String(error)));
  });
}

columnButtons.forEach(button => {
  button.addEventListener("click", async () => {
    if (!engineReady || gameOver || aiBusy) return;
    const column = Number(button.dataset.column);
    if (mode === "ai" && currentPlayer === aiPlayer) return;
    playColumn(column);
    if (mode === "ai" && !gameOver) await makeAIMove();
  });
});

modeButtons.forEach(button => {
  button.addEventListener("click", () => {
    if (mode === button.dataset.mode) return;
    mode = button.dataset.mode;
    modeButtons.forEach(item => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    startGame(true);
  });
});

document.querySelector("#new-game").addEventListener("click", () => startGame());
document.querySelector("#reset-scores").addEventListener("click", () => startGame(true));

startGame();
