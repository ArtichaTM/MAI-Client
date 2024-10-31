#include <vector>
#include <string>
#include <cassert>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "system.hpp"


TabSystem::TabSystem(float height, sf::Color color) : height(height), color(color) {
    font.loadFromFile(PATH_FONT_DEFAULT); // Ensure you have this font file
}

TabSystem::~TabSystem() {
    for (Tab*& tab : tabs) {
        delete tab;
    }
}

void TabSystem::handleEvent(const sf::Event& event) {
    for (size_t i = 0; i < tabs.size(); ++i) {
        Tab* tab = tabs[i];
        if (
            event.type == sf::Event::MouseButtonPressed
            &&
            tab->IsSwitcherInBounds(sf::Mouse::getPosition(*ROOT_WINDOW))
            &&
            event.mouseButton.button == sf::Mouse::Left
        ) {
            std::cout << "Active changed to "
                << (std::string)(tabs[i]->tabText.text.getString())
                << std::endl;
            active_tab_index = i;
        }
        tab->handleEvent(event);
    }
}

void TabSystem::draw(sf::RenderWindow& window) const {
    assert(!tabs.empty());
    for (size_t i = 0; i < tabs.size(); ++i) {
        Tab* tab = tabs[i];
        tab->draw(window, i == active_tab_index);
    }
}

Tab* TabSystem::addTab(const std::string& title) {
    float offset = 0;
    for (Tab*& tab : tabs) {
        offset += tab->tabText.shape.getGlobalBounds().width + offset;
    }
    Tab* tab = tabs.emplace_back(new Tab(title, offset, height, color));
    return tab;
}

Tab* TabSystem::operator[](const std::string &name)
{
    for (Tab*& tab : tabs) {
        if (tab->getName() == name)
            return tab;
    }
    throw std::range_error("Can't find this tab");
}

Tab* TabSystem::operator[](ushort index) { return tabs[index]; }
