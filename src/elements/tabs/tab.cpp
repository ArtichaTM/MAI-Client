#include "tab.hpp"

#include "config.hpp"

Tab::Tab(const std::string& title, float offset, float height) {
    tabText.setString(title);
    tabText.setCharacterSize(20);
    tabText.setFillColor(sf::Color::White);
    tabText.setPosition(offset, height);
}

void Tab::draw(sf::RenderWindow &window) const { draw(window, false); }

void Tab::draw(sf::RenderWindow& window, bool active) const {
    window.draw(tabText);
    for (const SFBase* element : elements) {
        window.draw(*element);
    }
}

void Tab::draw(sf::RenderTarget & target, sf::RenderStates states) const {
    throw std::logic_error("U can't call this method");
}

void Tab::handleEvent(const sf::Event& event) {
    for (SFBase* element : elements) {
        element->handleEvent(event);
    }
}

void Tab::setTabTextFont(const sf::Font &font) { tabText.setFont(font); }

bool Tab::IsInBounds(const sf::Vector2i& position) const {
    return tabText.getGlobalBounds().contains(position.x, position.y);
}

bool Tab::IsInBounds(const sf::Vector2f& position) const {
    return tabText.getGlobalBounds().contains(position);
}

void Tab::AddElement(SFBase* element) { elements.emplace_back(element); }
