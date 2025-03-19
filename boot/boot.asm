[BITS 16]             ; 16-bit Real Mode
[ORG 0x7C00]         ; Boot sector address

start:
    cli             ; Disable interrupts
    mov ax, 0x07C0
    mov ds, ax      ; Set data segment
    mov es, ax      ; Set extra segment

    mov si, msg     ; Load message address
    call print_string

    jmp $           ; Hang the system

; Function to print a string
print_string:
    mov ah, 0x0E    ; BIOS teletype function
.loop:
    lodsb           ; Load next byte from SI into AL
    cmp al, 0       ; Check for null terminator
    je done         ; If null, exit
    int 0x10        ; Print character to screen
    jmp .loop       ; Repeat

done:
    ret

msg db "Bootloader Loaded! Jumping to Kernel...", 0

; Boot sector padding (512 bytes total)
times 510-($-$$) db 0
dw 0xAA55         ; Boot sector signature
