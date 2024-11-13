#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "./system.hpp"
#include "system.hpp"


TabSystem::TabSystem(float height, sf::Color color) : height(height), color(color) {
    font.loadFromFile(PATH_FONT_DEFAULT); // Ensure you have this font file
}

TabSystem::~TabSystem() {
    for (Tab*& tab : tabs) {
        delete tab;
    }
}

TabSystem::operator std::string() const {
    std::stringstream ss;
    ss << "<TabSystem with " << tabs.size() << " tabs>";
    return ss.str();
}

void TabSystem::handleEvent(const sf::Event& event) {
    for (size_t i = 0; i < tabs.size(); ++i) {
        tabs[i]->handleEvent(event);
    }
}

void TabSystem::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    assert(!tabs.empty());
    for (size_t i = 0; i < tabs.size(); ++i) {
        Tab* tab = tabs[i];
        target.draw(*tab);
    }
}

sf::FloatRect TabSystem::getGlobalBounds() const
{
    sf::FloatRect rect(tabs[0]->getGlobalBounds());
    for (ushort i = 1; i < tabs.size(); i++) {
        rect.width += tabs[i]->getGlobalBounds().width;
    }
    return rect;
}

void TabSystem::move(const float left, const float top) {
    for (Tab*& tab : tabs) tab->move(left, top);
}

Tab* TabSystem::addTab(const std::string& title) {
    float offset = 0;
    for (Tab*& tab : tabs) {
        offset += tab->getGlobalBounds().width + offset;
    }
    Tab* tab = tabs.emplace_back(new Tab(title, offset, height, color));
    tab->tabText.setOnClick([this, tab](Button* button) {
        setActiveTab(tab);
    });
    if (tabs.size() == 1) {
        firstTabInit(tab);
    }
    return tab;
}

void TabSystem::firstTabInit(Tab* tab) {
    tab->setActive(true);
    active_tab = tab;
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

void TabSystem::setActiveTab(Tab *tab) {
    assert(active_tab);
    active_tab->setActive(false);
    active_tab = tab;
    tab->setActive(true);
}

void TabSystem::fit() {
    for (Tab*& tab : tabs) tab->tabText.fit();
}
