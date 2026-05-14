package mindustry.bridge;

import arc.*;
import arc.math.geom.Vec2;
import arc.util.*;
import arc.util.serialization.*;
import mindustry.*;
import mindustry.content.*;
import mindustry.game.EventType.*;
import mindustry.game.Rules;
import mindustry.gen.*;
import mindustry.maps.Map;
import mindustry.mod.*;
import mindustry.world.*;
import mindustry.world.blocks.distribution.Conveyor;
import mindustry.world.blocks.storage.CoreBlock;
import org.java_websocket.WebSocket;
import org.java_websocket.server.WebSocketServer;
import java.net.InetSocketAddress;
import java.util.concurrent.atomic.AtomicReference;

public class MindustryMod extends Mod {
    MindustryBridge server;

    @Override
    public void init() {
        server = new MindustryBridge();
        server.start();
        Log.info("MindustryBridge: WebSocket сервер запущен на порту 6789.");

        Events.on(GameOverEvent.class, event -> {
            Log.info("MindustryBridge: Game Over! Перезапуск через 3 секунды...");
            Time.runTask(180f, () -> {
                if (Vars.state.map != null) {
                    Vars.control.playMap(Vars.state.map, Vars.state.rules);
                }
            });
        });
    }
}

class MindustryBridge extends WebSocketServer {

    // ─── Блоки которые агент может строить ───
    static final Block[] BLOCKS = {
        Blocks.air,             // 0: Ничего / idle
        Blocks.mechanicalDrill, // 1: Бур (добывает медь)
        Blocks.conveyor,        // 2: Конвейер (транспорт)
        Blocks.copperWall,      // 3: Стена (защита)
        Blocks.duo,             // 4: Турель (атака)
    };

    // ─── Состояние между шагами ───
    private float  lastX, lastY;
    private int    lastCopper      = 0;
    private float  lastCoreHealth  = 1.0f;
    private int    stuckCounter    = 0;
    private static final int STUCK_THRESHOLD = 3; // шагов подряд без движения

    // ─── WebSocket клиент (один агент) ───
    private volatile WebSocket agentConn = null;

    // ─── Защита от race condition: пишем state атомарно ───
    private final AtomicReference<String> pendingState = new AtomicReference<>(null);

    public MindustryBridge() {
        super(new InetSocketAddress(6789));
        setReuseAddr(true);
    }

    // ════════════════════════════════════════
    //  Входящее сообщение от агента
    // ════════════════════════════════════════
    @Override
    public void onMessage(WebSocket conn, String message) {
        try {
            JsonValue data = new JsonReader().parse(message);

            if (data.has("reset")) {
                // Перезапуск карты — выполнить на главном потоке Arc
                Core.app.post(() -> {
                    Map map = Vars.state.map;
                    Rules rules = Vars.state.rules;
                    if (map != null) {
                        Vars.control.playMap(map, rules);
                        resetState();
                        sendStateWhenReady(conn, 60);
                    } else {
                        conn.send(doneJson());
                    }
                });
                // Сразу отвечаем "done" чтобы агент не завис
                return;
            }

            // ─── Читаем команду ───
            final float   vx     = data.getFloat("vx", 0f)  * 15f;
            final float   vy     = data.getFloat("vy", 0f)  * 15f;
            final int     bType  = data.getInt("type", 0);
            final int     relX   = data.getInt("x", 0);
            final int     relY   = data.getInt("y", 0);
            final boolean delete = data.getBoolean("delete", false);

            // ─── Применяем на главном потоке игры ───
            Core.app.post(() -> {
                Unit unit = Vars.player.unit();
                if (unit == null || Vars.state.isPaused() || Vars.state.gameOver) return;

                // Движение
                unit.moveAt(new Vec2(vx, vy));

                // Постройка / удаление блока
                int tx = relX + unit.tileX() - 16;
                int ty = relY + unit.tileY() - 16;
                Tile tile = Vars.world.tile(tx, ty);

                if (tile != null && !(tile.block() instanceof CoreBlock)) {
                    if (delete) {
                        tile.setNet(Blocks.air);
                        Fx.breakBlock.at(tile.drawx(), tile.drawy(), 1f);

                    } else if (bType > 0 && bType < BLOCKS.length) {
                        // Разрешаем строить только не вплотную к себе (избегаем застревания)
                        boolean farEnough = Math.abs(tx - unit.tileX()) > 1
                                         || Math.abs(ty - unit.tileY()) > 1;
                        if (farEnough && tile.block() == Blocks.air) {
                            int rotation = (BLOCKS[bType] == Blocks.conveyor) ? rotationTowardCore(tx, ty) : 0;
                            tile.setNet(BLOCKS[bType], Vars.player.team(), rotation);
                            Fx.placeBlock.at(tile.drawx(), tile.drawy(), BLOCKS[bType].size);
                        }
                    }
                }
            });

            // ─── Возвращаем состояние (строим на том же потоке WS, snapshot safe) ───
            // Небольшая задержка чтобы Core.app.post успел выполниться
            Time.runTask(2f, () -> {
                String state = buildStateJson();
                conn.send(state);
            });

        } catch (Exception e) {
            Log.err("MindustryBridge.onMessage", e);
            conn.send(doneJson());
        }
    }

    // ════════════════════════════════════════
    //  Построение JSON состояния
    // ════════════════════════════════════════
    private void sendStateWhenReady(WebSocket conn, int attemptsLeft) {
        Time.runTask(10f, () -> {
            if (isGameReady()) {
                conn.send(buildStateJson());
            } else if (attemptsLeft > 0) {
                sendStateWhenReady(conn, attemptsLeft - 1);
            } else {
                conn.send(doneJson());
            }
        });
    }

    private boolean isGameReady() {
        return Vars.player != null
            && Vars.player.unit() != null
            && Vars.state.map != null
            && Vars.state.isPlaying()
            && !Vars.state.gameOver;
    }

    private String buildStateJson() {
        Unit u = Vars.player.unit();
        if (!isGameReady()) return doneJson();

        Building core = Vars.player.team().core();

        // ── Медь ──
        int currentCopper = (core != null) ? core.items.get(Items.copper) : 0;
        int deltaCopper   = Math.max(0, currentCopper - lastCopper);
        lastCopper = currentCopper;

        // ── Застревание (несколько шагов без движения = stuck) ──
        boolean movedThisStep = (Math.abs(u.x - lastX) > 0.5f || Math.abs(u.y - lastY) > 0.5f);
        if (movedThisStep) { stuckCounter = 0; }
        else               { stuckCounter++; }
        boolean isStuck = stuckCounter >= STUCK_THRESHOLD;
        lastX = u.x;
        lastY = u.y;

        // ── Здоровье ядра ──
        float coreHealthNorm = (core != null) ? (core.health / core.maxHealth) : 0f;

        // ── Вектор к ядру ──
        float coreDx = (core != null) ? clamp((core.tileX() - u.tileX()) / 25f, -1f, 1f) : 0f;
        float coreDy = (core != null) ? clamp((core.tileY() - u.tileY()) / 25f, -1f, 1f) : 0f;

        // ── Вид вокруг агента 32×32 ──
        int sx = u.tileX() - 16;
        int sy = u.tileY() - 16;

        // Каналы карты (11 штук):
        //  0 - тип пола (floor id / 255)
        //  1 - медная руда
        //  2 - ядро
        //  3 - бур
        //  4 - конвейер
        //  5 - стена (любая)
        //  6 - враги (нормированное кол-во)
        //  7 - здоровье блока (0..1)
        //  8 - направление конвейера (0..3 / 4)
        //  9 - позиция агента (центр карты = 1)
        // 10 - кол-во предметов в конкретной ячейке / 100
        float[][][] map = new float[11][32][32];

        // Позиция агента всегда в центре
        map[9][15][15] = 1f;
        map[9][16][16] = 1f;

        float nearestCopperDist = 32f;
        int drillCount = 0;

        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                Tile t = Vars.world.tile(sx + x, sy + y);
                if (t == null) continue;

                map[0][y][x] = t.floor().id / 255f;

                boolean hasCopper = (t.overlay() == Blocks.oreCopper);
                map[1][y][x] = hasCopper ? 1f : 0f;
                if (hasCopper) {
                    float dist = (float) Math.sqrt((x - 16) * (x - 16) + (y - 16) * (y - 16));
                    if (dist < nearestCopperDist) nearestCopperDist = dist;
                }

                map[2][y][x] = (t.block() instanceof CoreBlock) ? 1f : 0f;

                boolean hasDrill = (t.block() == Blocks.mechanicalDrill);
                map[3][y][x] = hasDrill ? 1f : 0f;
                if (hasDrill) drillCount++;

                boolean hasConveyor = (t.block() instanceof Conveyor);
                map[4][y][x] = hasConveyor ? 1f : 0f;

                // Стена (любой wall-блок)
                map[5][y][x] = t.block().isStatic() ? 1f : 0f;

                // Здоровье блока
                Building b = t.build;
                if (b != null) {
                    map[7][y][x] = b.health / b.maxHealth;
                    // Предметы в здании (напр. в хранилище)
                    map[10][y][x] = Math.min(1f, b.items != null ? b.items.total() / 100f : 0f);
                }

                // Направление конвейера
                if (hasConveyor) {
                    map[8][y][x] = t.build != null ? (t.build.rotation / 4f) : 0f;
                }
            }
        }

        // ── Враги поблизости ──
        int enemyCount = 0;
        for (Unit enemy : Groups.unit) {
            if (enemy.team != Vars.player.team()) {
                float dx = enemy.x - u.x;
                float dy = enemy.y - u.y;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                if (dist < 32 * 8) { // 32 тайла * 8 пикс/тайл
                    enemyCount++;
                    // Отмечаем позицию врага на карте (канал 6)
                    int ex = (int)((enemy.x - u.x) / 8f) + 16;
                    int ey = (int)((enemy.y - u.y) / 8f) + 16;
                    if (ex >= 0 && ex < 32 && ey >= 0 && ey < 32) {
                        map[6][ey][ex] = Math.min(1f, map[6][ey][ex] + 0.5f);
                    }
                }
            }
        }

        // ── Скаляры (16 штук) ──
        float[] scalars = new float[16];
        scalars[0]  = u.healthf();                           // здоровье юнита
        scalars[1]  = (float) u.tileX() / 100f;             // X позиция
        scalars[2]  = (float) u.tileY() / 100f;             // Y позиция
        scalars[3]  = (float) u.stack.amount / 50f;         // предметы в руках
        scalars[4]  = (float) deltaCopper / 20f;            // медь за шаг
        scalars[5]  = isStuck ? 1f : 0f;                    // застрял?
        scalars[6]  = coreDx;                               // направление к ядру X
        scalars[7]  = coreDy;                               // направление к ядру Y
        scalars[8]  = nearestCopperDist / 32f;              // расстояние до меди
        scalars[9]  = Math.min(1f, drillCount / 20f);       // кол-во буров
        scalars[10] = Math.min(1f, enemyCount / 10f);       // кол-во врагов
        scalars[11] = coreHealthNorm;                        // здоровье ядра
        int edgeDist = Math.min(Math.min(u.tileX(), u.tileY()),
            Math.min(Vars.world.width() - 1 - u.tileX(), Vars.world.height() - 1 - u.tileY()));
        scalars[12] = clamp((8f - edgeDist) / 8f, 0f, 1f);
        // 12-15: резерв (нули)

        return String.format(
            "{\"map\":%s,\"scalars\":%s,\"done\":%b,\"ready\":true}",
            mapToJson(map),
            scalarsToJson(scalars),
            Vars.state.gameOver
        );
    }

    // ════════════════════════════════════════
    //  Вспомогательные методы
    // ════════════════════════════════════════

    private void resetState() {
        lastX = 0; lastY = 0;
        lastCopper = 0;
        lastCoreHealth = 1.0f;
        stuckCounter = 0;
    }

    private int rotationTowardCore(int tx, int ty) {
        Building core = Vars.player.team().core();
        if (core == null) return 0;

        int dx = core.tileX() - tx;
        int dy = core.tileY() - ty;
        if (Math.abs(dx) >= Math.abs(dy)) {
            return dx >= 0 ? 0 : 2;
        }
        return dy >= 0 ? 1 : 3;
    }

    private String doneJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"map\":[],\"scalars\":[");
        for (int i = 0; i < 16; i++) { sb.append("0"); if (i < 15) sb.append(","); }
        sb.append("],\"done\":true,\"ready\":false}");
        return sb.toString();
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    private String scalarsToJson(float[] s) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < s.length; i++) {
            sb.append(s[i]);
            if (i < s.length - 1) sb.append(",");
        }
        return sb.append("]").toString();
    }

    private String mapToJson(float[][][] map) {
        int C = map.length, H = map[0].length, W = map[0][0].length;
        StringBuilder sb = new StringBuilder("[");
        for (int c = 0; c < C; c++) {
            sb.append("[");
            for (int y = 0; y < H; y++) {
                sb.append("[");
                for (int x = 0; x < W; x++) {
                    // Округляем до 4 знаков чтобы уменьшить размер JSON
                    sb.append(Math.round(map[c][y][x] * 10000) / 10000f);
                    if (x < W - 1) sb.append(",");
                }
                sb.append("]");
                if (y < H - 1) sb.append(",");
            }
            sb.append("]");
            if (c < C - 1) sb.append(",");
        }
        return sb.append("]").toString();
    }

    // ════════════════════════════════════════
    //  WebSocket lifecycle
    // ════════════════════════════════════════

    @Override
    public void onOpen(WebSocket conn, org.java_websocket.handshake.ClientHandshake h) {
        agentConn = conn;
        Log.info("MindustryBridge: Агент подключён с " + conn.getRemoteSocketAddress());
    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        agentConn = null;
        Log.info("MindustryBridge: Агент отключён (причина: " + reason + ")");
    }

    @Override
    public void onError(WebSocket conn, Exception ex) {
        Log.err("MindustryBridge WebSocket ошибка", ex);
    }

    @Override
    public void onStart() {
        Log.info("MindustryBridge: WebSocketServer запущен.");
    }
}
