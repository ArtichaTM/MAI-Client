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

sf::FloatRect TabSystem::getGlobalBounds() const {
    if (tabs.empty()) return sf::FloatRect();
    sf::FloatRect left_bounds = tabs[0]->getGlobalBounds();
    sf::FloatRect right_bounds = tabs[tabs.size()-1]->getGlobalBounds();
    left_bounds.width = right_bounds.left + right_bounds.width;
    return left_bounds;
}
TabSystem* TabSystem::setLeft(float value) {
    throw std::logic_error("TabSystem is absolute");
}
TabSystem* TabSystem::setTop(float value) {
    throw std::logic_error("TabSystem is absolute");
}
TabSystem* TabSystem::setWidth(float value) {
    throw std::logic_error("TabSystem is absolute");
}
TabSystem* TabSystem::setHeight(float value) {
    throw std::logic_error("TabSystem is absolute");
}
float TabSystem::getLeft() const {
    return tabs[0]->getLeft();
}
float TabSystem::getTop() const {
    return tabs[0]->getTop();
}
float TabSystem::getWidth() const {
    Tab* left_tab = tabs[tabs.size()-1];
    return getLeft() - (
        left_tab->getLeft() + left_tab->getWidth()
    );
}
float TabSystem::getHeight() const {
    return height;
}

Tab* TabSystem::addTab(const std::string& title) {
    float offset = 0;
    for (Tab*& tab : tabs) {
        offset += tab->tabText.shape.getGlobalBounds().width + offset;
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
