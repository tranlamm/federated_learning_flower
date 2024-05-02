const DIRECTION_CONST = {
    UP: 1,
    DOWN: 2,
    LEFT: 3,
    RIGHT: 4
}

const MODE = {
    EASE: 1,
    MEDIUM: 2,
    HARD: 3,
    EXCLUSIVE: 4
}

function getRandomNumber(min, max) {
    const randomNumber = Math.floor(Math.random() * (max - min + 1)) + min;
    return randomNumber;
}

let map, g_x, g_y, g_curX, g_curY;
let g_mapCame = {};
let g_mapCameMainPath = {};
let g_listPath = [];
let prevDirection = null;

let createMap = (dimension) => {
    map = Array.from({ length: dimension }, () => Array(dimension).fill(0));
    for (let i = 0; i < dimension; ++i) {
        map[i][0] = 1;
        map[i][dimension - 1] = 1;
        map[0][i] = 1;
        map[dimension - 1][i] = 1;
    }
}

let getNumAvailableSlotDirection = (direction, x, y) => {
    let cur = {x: x, y: y};
    let max = 0;
    let listValidBlockPosition = [];
    switch (direction) {
        case DIRECTION_CONST.UP:
            for (let i = cur.y - 1; i > -1; --i) {
                if (map[i][cur.x] == 1) {
                    listValidBlockPosition.push({x: cur.x, y: i});
                    break;
                }
                if (g_mapCame[cur.x + "_" + i] != true) {
                    listValidBlockPosition.push({x: cur.x, y: i});
                }
                max++;
            }
            break;
        case DIRECTION_CONST.DOWN:
            for (let i = cur.y + 1; i < map.length; ++i) {
                if (map[i][cur.x] == 1) {
                    listValidBlockPosition.push({x: cur.x, y: i});
                    break;
                }
                if (g_mapCame[cur.x + "_" + i] != true) {
                    listValidBlockPosition.push({x: cur.x, y: i});
                }
                max++;
            }
            break;
        case DIRECTION_CONST.LEFT:
            for (let i = cur.x - 1; i > -1; --i) {
                if (map[cur.y][i] == 1) {
                    listValidBlockPosition.push({x: i, y: cur.y});
                    break;
                }
                if (g_mapCame[i + "_" + cur.y] != true) {
                    listValidBlockPosition.push({x: i, y: cur.y});
                }
                max++;
            }
            break;
        case DIRECTION_CONST.RIGHT:
            for (let i = cur.x + 1; i < map[0].length; ++i) {
                if (map[cur.y][i] == 1) {
                    listValidBlockPosition.push({x: i, y: cur.y});
                    break;
                }
                if (g_mapCame[i + "_" + cur.y] != true) {
                    listValidBlockPosition.push({x: i, y: cur.y});
                }
                max++;
            }
            break;
    }
    return {max: max, list: listValidBlockPosition};
}

let getOppositeDirection = (dir) => {
    switch (dir) {
        case DIRECTION_CONST.UP:
            return DIRECTION_CONST.DOWN;
        case DIRECTION_CONST.DOWN:
            return DIRECTION_CONST.UP;
        case DIRECTION_CONST.LEFT:
            return DIRECTION_CONST.RIGHT;
        case DIRECTION_CONST.RIGHT:
            return DIRECTION_CONST.LEFT;
    }
}

let genMainPath = (numTurn) => {
    for (let turn = 0; turn < numTurn; ++turn) {
        let poolDir = [DIRECTION_CONST.UP, DIRECTION_CONST.DOWN, DIRECTION_CONST.RIGHT, DIRECTION_CONST.LEFT];
        if (prevDirection != null) {
            poolDir = poolDir.filter((ele) => ele != getOppositeDirection(prevDirection));
        }
        let found = false;
        let path;
        let dir;
        while (!found) {
            if (poolDir.length == 0) {
                console.log("not path");
                break;
            }
            let rand = getRandomNumber(0, poolDir.length-1);
            dir = poolDir[rand];
            path = getNumAvailableSlotDirection(dir, g_curX, g_curY);
            if (path.max != 0) {
                found = true;
            } else {
                poolDir.splice(rand, 1);
            }
        }
        if (!found) {
            dir = getOppositeDirection(prevDirection);
            path = getNumAvailableSlotDirection(dir, g_curX, g_curY);
        }
        prevDirection = dir;
        let listBlock = path.list;
        if (listBlock.length == 0) {
            let prev = g_listPath[g_listPath.length - 2];
            g_curX = prev.x;
            g_curY = prev.y;
            continue;
        }

        let rand = getRandomNumber(0, listBlock.length - 1);
        let block = listBlock[rand];
        map[block.y][block.x] = 1;
        switch (dir) {
            case DIRECTION_CONST.UP:
                for (let i = g_curY - 1; i > block.y; --i) {
                    g_mapCame[g_curX + "_" + i] = true;
                    g_mapCameMainPath[g_curX + "_" + i] = true;
                }
                g_curY = block.y + 1;
                break;
            case DIRECTION_CONST.DOWN:
                for (let i = g_curY + 1; i < block.y; ++i) {
                    g_mapCame[g_curX + "_" + i] = true;
                    g_mapCameMainPath[g_curX + "_" + i] = true;
                }
                g_curY = block.y - 1;
                break;
            case DIRECTION_CONST.LEFT:
                for (let i = g_curX - 1; i > block.x; --i) {
                    g_mapCame[i + "_" + g_curY] = true;
                    g_mapCameMainPath[i + "_" + g_curY] = true;
                }
                g_curX = block.x + 1;
                break;
            case DIRECTION_CONST.RIGHT:
                for (let i = g_curX + 1; i < block.x; ++i) {
                    g_mapCame[i + "_" + g_curY] = true;
                    g_mapCameMainPath[i + "_" + g_curY] = true;
                }
                g_curX = block.x - 1;
                break;
        }
        g_listPath.push({x: g_curX, y: g_curY});
    }
}

let generateFakePath = (turn, prevDir, x, y) => {
    if (turn < 0) return;
    let poolDir = [DIRECTION_CONST.UP, DIRECTION_CONST.DOWN, DIRECTION_CONST.RIGHT, DIRECTION_CONST.LEFT];
    for (let i = 0; i < poolDir.length; ++i) {
        let dir = poolDir[i];
        if (prevDir != null) {
            if (dir == getOppositeDirection(prevDir)) continue;
        }
        let path = getNumAvailableSlotDirection(dir, x, y);
        if (path.max == 0) continue;
        let listBlock = path.list;
        let rand = getRandomNumber(0, listBlock.length - 1);
        
        let block = listBlock[rand];
        map[block.y][block.x] = 1;
        switch (dir) {
            case DIRECTION_CONST.UP:
                for (let i = y - 1; i > block.y; --i) {
                    g_mapCame[x + "_" + i] = true;
                }
                generateFakePath(turn - 1, dir, x, block.y + 1);
                break;
            case DIRECTION_CONST.DOWN:
                for (let i = y + 1; i < block.y; ++i) {
                    g_mapCame[x + "_" + i] = true;
                }
                generateFakePath(turn - 1, dir, x, block.y - 1);
                break;
            case DIRECTION_CONST.LEFT:
                for (let i = x - 1; i > block.x; --i) {
                    g_mapCame[i + "_" + y] = true;
                }
                generateFakePath(turn - 1, dir, block.x + 1, y);
                break;
            case DIRECTION_CONST.RIGHT:
                for (let i = x + 1; i < block.x; ++i) {
                    g_mapCame[i + "_" + y] = true;
                }
                generateFakePath(turn - 1, dir, block.x - 1, y);
                break;
        }
    }
}

let generateBot = (numBot) => {
    let hash_map = {};
    let hash_list = [];
    for (let i = 0; i < map.length; ++i) {
        for (let j = 0; j < map[0].length; ++j) {
            if (map[i][j] == 0 && g_mapCameMainPath[j + "_" + i] != true) {
                hash_map[j + "_" + i] = {x: j, y: i};    
            } else {
                hash_map[j + "_" + i] = false;
            }
        }
    }
    hash_map[g_x + "_" + g_y] = false;
    let poolDir = [DIRECTION_CONST.UP, DIRECTION_CONST.DOWN, DIRECTION_CONST.RIGHT, DIRECTION_CONST.LEFT];
    for (let i = 0; i < poolDir.length; ++i) {
        let dir = poolDir[i];
        let num = getNumAvailableSlotDirection(dir, g_x, g_y).max;
        switch (dir) {
            case DIRECTION_CONST.UP:
                for (let i = 1; i <= num; ++i) {
                    hash_map[g_x + "_" + (g_y - i)] = false;
                }
                break;
            case DIRECTION_CONST.DOWN:
                for (let i = 1; i <= num; ++i) {
                    hash_map[g_x + "_" + (g_y + i)] = false;
                }
                break;
            case DIRECTION_CONST.LEFT:
                for (let i = 1; i <= num; ++i) {
                    hash_map[(g_x - i) + "_" + g_y] = false;
                }
                break;
            case DIRECTION_CONST.RIGHT:
                for (let i = 1; i <= num; ++i) {
                    hash_map[(g_x + i) + "_" + g_y] = false;
                }
                break;
        }
    }
    for (let slot in hash_map) {
        if (hash_map[slot] != false) {
            hash_list.push(hash_map[slot]);
        }
    }
    for (let i = 0; i < numBot; ++i) {
        let idx = getRandomNumber(0, hash_list.length - 1);
        let slot = hash_list[idx];
        map[slot.y][slot.x] = 2;
    }
}

let printJSONMap = () => {
    let string = "[\n";
    let end = g_listPath[g_listPath.length - 1];
    map[end.y][end.x] = 3;
    for (let i = 0; i < map.length; ++i) {
        string += "[";
        let length = map[i].length;
        for (let j = 0; j < length; ++j) {
            string += map[i][j] + (j == length - 1 ? "" : ", ");
        }
        string += (i == length - 1 ? "]\n" : "],\n");
    }
    string += "],";
    console.log(string);
}

let printPath = () => {
    let string = "";
    for (let i = 0; i < g_listPath.length; ++i) {
        let slot = g_listPath[i];
        string += "[" + slot.x + "," + slot.y + "]  ";
    }
    console.log(string);
}

let genMap = (x, y, dimension, numBot) => {
    g_x = x;
    g_y = y;
    g_curX = x;
    g_curY = y;
    g_mapCame[g_curX + "_" + g_curY] = true;
    g_mapCameMainPath[g_curX + "_" + g_curY] = true;
    g_listPath.push({x: g_x, y: g_y});
    createMap(dimension);
    genMainPath(getRandomNumber(7, 12));
    generateFakePath(getRandomNumber(7, 12), null, g_x, g_y);
    // generateBot(numBot);
}

let autoGenMap = (mode) => {
    let dimension;
    let numBot;
    switch (mode) {
        case MODE.EASE:
            numBot = getRandomNumber(1, 3);
            dimension = getRandomNumber(12, 15);
            break;
        case MODE.MEDIUM:
            numBot = getRandomNumber(2, 4);
            dimension = getRandomNumber(15, 25);
            break;
        case MODE.HARD:
            numBot = getRandomNumber(3, 6);
            dimension = getRandomNumber(25, 35);
            break;
        default:
            numBot = getRandomNumber(4, 8);
            dimension = getRandomNumber(35, 45);
            break;
    }
    let half = Math.floor(dimension / 2);
    let x = getRandomNumber(half - 4, half + 4);
    let y = getRandomNumber(half - 4, half + 4);
    genMap(x, y, dimension, numBot);
    console.log(dimension + " " + x + " " + (dimension - 1 - y));
}

autoGenMap(MODE.EXCLUSIVE);
printJSONMap();
printPath();