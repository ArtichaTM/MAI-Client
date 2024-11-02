#include "./header.hpp"

TabSystem* build_ui() {
    TabSystem* tabsys = new TabSystem(10.f, sf::Color::Green);
    add_overview(tabsys);
    add_ai_modules_influences(tabsys);
    return tabsys;
}
