#include <windows.h>
#include <string>


static const bool enable_VT_mode(void) {
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut == INVALID_HANDLE_VALUE) return false;

	DWORD dwMode = 0;
	if (!GetConsoleMode(hOut, &dwMode)) return false;

	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(hOut, dwMode)) return false;

	return true;
}


const std::string get_terminator(void) noexcept {
	if (enable_VT_mode()) return "\x1b[0K";
	else return "                        ";
}