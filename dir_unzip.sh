function main() {
    local dir="$1"
    echo "Unzipping files in directory: $dir"
    echo ""

    if [[ ! -d "$dir" ]]; then
        echo "Error: '$zip_file' is not a directory."
        return 1
    fi

    for item in "$dir"/*; do
        if [[ -d "$item" ]]; then
            echo "Skipping directory: $item"
            continue
        fi

        if [[ "$item" == *.zip ]]; then
            unzip -q "$item" -d "${item%.zip}"
            echo "Unzipped: $item"
        else
            echo "Skipping non-zip file: $item"
        fi
    done
}

main "$@"
