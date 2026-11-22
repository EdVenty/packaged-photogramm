#!/usr/bin/env bash
# Interactive installer: choose modules, configure .env, build and start the stack.
#
#   ./install.sh                         interactive
#   ./install.sh -y --modules mesh,splat --water-splatting no
#   ./install.sh --modules all --no-start
#
# Modules (docker compose profiles): core is always installed.
#   mesh    OpenMVS textured mesh                    (worker-mvs)
#   splat   gaussian splatting, nerfstudio            (worker-splat)
#   refrac  Refractive COLMAP for underwater housings (worker-sfm-refrac)
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE=.env
OPTIONAL=(mesh splat refrac)
declare -A SERVICE=([mesh]=worker-mvs [splat]=worker-splat [refrac]=worker-sfm-refrac)
declare -A TITLE=(
  [mesh]="Меш (OpenMVS: плотное облако, меш, текстура)"
  [splat]="Gaussian Splatting (nerfstudio splatfacto)"
  [refrac]="Refractive COLMAP — подводные корпуса flat/dome port (долгая сборка)"
)

ASSUME_YES=0 BUILD=1 START=1 MODULES_ARG="" WS_ARG="" TOKEN_ARG=""

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1 ;;
    --modules) MODULES_ARG="$2"; shift ;;
    --water-splatting) WS_ARG="$2"; shift ;;
    --token) TOKEN_ARG="$2"; shift ;;
    --no-build) BUILD=0 ;;
    --no-start) START=0 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

c_bold=$'\e[1m' c_dim=$'\e[2m' c_grn=$'\e[32m' c_ylw=$'\e[33m' c_red=$'\e[31m' c_off=$'\e[0m'
info() { echo "${c_grn}==>${c_off} $*"; }
warn() { echo "${c_ylw}warning:${c_off} $*" >&2; }
die()  { echo "${c_red}error:${c_off} $*" >&2; exit 1; }

# ------------------------------------------------------------------ .env helpers
[[ -f $ENV_FILE ]] || cp .env.example "$ENV_FILE"

get_env() { # key -> value from .env (empty if absent)
  awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); v=$0} END {print v}' "$ENV_FILE"
}
set_env() { # key value -> replace or append in .env
  local tmp; tmp=$(mktemp)
  awk -v k="$1" -v v="$2" 'BEGIN{done=0} $0 ~ "^"k"=" {print k"="v; done=1; next} {print}
       END{if(!done) print k"="v}' "$ENV_FILE" > "$tmp"
  cat "$tmp" > "$ENV_FILE"; rm -f "$tmp"
}

ask_yn() { # question default(y|n) -> exit status
  local q=$1 def=$2 a
  if [[ $ASSUME_YES == 1 ]]; then [[ $def == y ]]; return; fi
  local hint="[Y/n]"; [[ $def == n ]] && hint="[y/N]"
  read -r -p "$q $hint " a || true
  a=${a:-$def}
  [[ $a =~ ^[YyДд] ]]
}
ask_str() { # question default -> echo answer
  local q=$1 def=$2 a
  if [[ $ASSUME_YES == 1 ]]; then echo "$def"; return; fi
  read -r -p "$q [${def}]: " a || true
  echo "${a:-$def}"
}

# ------------------------------------------------------------------ prerequisites
command -v docker >/dev/null || die "docker не найден"
docker compose version >/dev/null 2>&1 || die "нужен плагин docker compose v2"
if ! docker info 2>/dev/null | grep -qi 'nvidia'; then
  warn "в docker не найден runtime nvidia — GPU-модули не запустятся без NVIDIA Container Toolkit"
fi

# ------------------------------------------------------------------ module selection
current=$(get_env COMPOSE_PROFILES)
declare -A SEL=()
if [[ -n $MODULES_ARG ]]; then
  case "$MODULES_ARG" in
    all) for m in "${OPTIONAL[@]}"; do SEL[$m]=1; done ;;
    none|core) ;;
    *) IFS=, read -r -a picked <<<"$MODULES_ARG"
       for m in "${picked[@]}"; do
         [[ -n ${SERVICE[$m]:-} ]] || die "неизвестный модуль: $m (mesh, splat, refrac, all, none)"
         SEL[$m]=1
       done ;;
  esac
else
  echo "${c_bold}Выбор модулей${c_off} ${c_dim}(ядро — API, предобработка, COLMAP SfM — ставится всегда)${c_off}"
  for m in "${OPTIONAL[@]}"; do
    def=n
    if [[ ",$current," == *",$m,"* || ",$current," == *",all,"* ]]; then def=y; fi
    [[ -z $current && $m != refrac ]] && def=y
    if ask_yn "  ${TITLE[$m]}?" "$def"; then SEL[$m]=1; fi
  done
fi

profiles=""
for m in "${OPTIONAL[@]}"; do [[ -n ${SEL[$m]:-} ]] && profiles+="${profiles:+,}$m"; done
set_env COMPOSE_PROFILES "$profiles"

if [[ -n ${SEL[splat]:-} ]]; then
  if [[ -n $WS_ARG ]]; then
    [[ $WS_ARG =~ ^(1|y|yes|true)$ ]] && ws=1 || ws=0
  else
    curws=$(get_env INSTALL_WATERSPLATTING); def=y; [[ $curws == 0 ]] && def=n
    if ask_yn "  + WaterSplatting (подводные сплаты)?" "$def"; then ws=1; else ws=0; fi
  fi
  set_env INSTALL_WATERSPLATTING "$ws"
fi

# ------------------------------------------------------------------ GPU architecture
if command -v nvidia-smi >/dev/null 2>&1; then
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr -d ' ' || true)
  if [[ -n $caps ]]; then
    cuda_arch=$(echo "$caps" | tr -d '.' | paste -sd';' -)
    torch_arch=$(echo "$caps" | paste -sd';' -)
    if ask_yn "Найдены GPU с compute capability ${caps//$'\n'/, }. Собирать только под них (быстрее)?" y; then
      set_env CUDA_ARCHITECTURES "$cuda_arch"
      set_env TORCH_CUDA_ARCH_LIST "$torch_arch"
    fi
  fi
fi

# ------------------------------------------------------------------ API token
token=${TOKEN_ARG:-$(get_env PG_API_TOKEN)}
if [[ -z $token && -z $TOKEN_ARG ]] && ask_yn "Сгенерировать API-токен (рекомендуется, если порт доступен из сети)?" y; then
  token=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
fi
set_env PG_API_TOKEN "$token"
port=$(get_env PG_PORT); port=${port:-8000}

# ------------------------------------------------------------------ summary
echo
echo "${c_bold}Конфигурация (${ENV_FILE})${c_off}"
echo "  модули:          core${profiles:+,$profiles}"
[[ -n ${SEL[splat]:-} ]] && echo "  WaterSplatting:  $([[ $(get_env INSTALL_WATERSPLATTING) == 1 ]] && echo да || echo нет)"
echo "  CUDA arch:       $(get_env CUDA_ARCHITECTURES)"
echo "  API:             http://<host>:${port}  токен: ${token:-<нет>}"
echo

# Stop services of modules that were deselected.
for m in "${OPTIONAL[@]}"; do
  if [[ -z ${SEL[$m]:-} ]]; then
    COMPOSE_PROFILES=all docker compose rm -sf "${SERVICE[$m]}" >/dev/null 2>&1 || true
  fi
done

if [[ $BUILD == 1 ]]; then
  info "сборка образов (первый раз может занять 30–90 минут)…"
  docker compose build
fi
if [[ $START == 1 ]]; then
  info "запуск…"
  docker compose up -d
  docker compose ps
  echo
  info "готово. TUI: pip install ./tui && pgtui --server http://localhost:${port}${token:+ --token $token}"
  info "или прямо здесь: docker compose run --rm tui"
fi
