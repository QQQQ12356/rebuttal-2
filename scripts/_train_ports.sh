# Reserve four consecutive defaults per task. A caller may override MASTER_PORT.
set_training_port() {
    local base_port=$1 offset
    case "$METHOD" in
        seco) offset=0 ;;
        comi) offset=1 ;;
        icae) offset=2 ;;
        ram)  offset=3 ;;
        *) printf 'Unknown compression method: %s\n' "$METHOD" >&2; return 2 ;;
    esac
    MASTER_PORT="${MASTER_PORT:-$((base_port + offset))}"
}
