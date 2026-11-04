# Rendered with VHS: https://github.com/charmbracelet/vhs
#   vhs demo.tape
# Edit the Type lines below to show what this project is actually good at.
# A demo that shows one real task end to end beats one that shows six flags.

Output assets/demo.gif

Require cargo

Set Shell "bash"
Set FontSize 22
Set Width 1200
Set Height 640
Set Padding 20
Set Margin 30
Set MarginFill "#1e1b2e"
Set BorderRadius 10
Set WindowBar Colorful
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms

# --- setup, not recorded ---
Hide
Type "clear"
Enter
Show

Type "cargo run -- --help"
Sleep 400ms
Enter
Sleep 5s

# --- hold the final frame so the loop reads well ---
Sleep 3s
