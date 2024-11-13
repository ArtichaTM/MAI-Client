#pragma once
#include <vector>
#include <string>

#include <SFML/Graphics.hpp>

#include "./tab.hpp"
#include "config.hpp"
#include "elements/base.hpp"
#include "elements/list/horizontal.hpp"

class TabSystem : public SFBase {
    HorizontalList* tabs;
    // std::vector<Tab*> tabs;
    Tab* active_tab;
    sf::Font font;
    const float height;
    const sf::Color color;
    static constexpr float offset = 50.f;

public:
    TabSystem(float height, sf::Color color);
    ~TabSystem();
    operator std::string() const;

    void handleEvent(const sf::Event&) override;
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    sf::FloatRect getGlobalBounds() const override;
    void move(const float left, const float top) override;
    void fit() override;

    Tab* addTab(const std::string& title);

    void setActiveTab(Tab* tab);

    Tab* operator[](const std::string& name);
    Tab* operator[](ushort index);
};
