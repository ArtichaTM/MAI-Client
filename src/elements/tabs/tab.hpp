#pragma once
#include <vector>

#include <SFML/Graphics.hpp>

#include "../base.hpp"

class Tab : public SFBase {
    sf::Text tabText;
    std::vector<SFBase*> elements;
public:
    Tab(const std::string& title, float offset, float height);
    void draw(sf::RenderWindow& window) const override;
    void draw(sf::RenderWindow& window, bool active = false) const;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void handleEvent(const sf::Event& event) override;
    void setTabTextFont(const sf::Font& font);
    bool IsInBounds(const sf::Vector2i& position) const;
    bool IsInBounds(const sf::Vector2f& position) const;
    void AddElement(SFBase* element);
};
